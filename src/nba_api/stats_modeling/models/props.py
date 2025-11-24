"""
Player Props Models for Sports Betting

Predicts player prop outcomes (over/under) for any statistic.
Produces calibrated probability estimates for betting decisions.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
import warnings

from nba_api.stats_modeling.models.base import NBABaseModel
from nba_api.stats_modeling.utils.features import FeatureEngineer


class PlayerPropsPredictor(NBABaseModel):
    """
    Predicts player prop outcomes for any statistic.

    Supports:
    - Points, rebounds, assists, steals, blocks
    - Three-pointers made, free throws made
    - Combo props (PTS+REB, PTS+AST, PTS+REB+AST)
    - Any custom stat combination

    Betting Applications:
    - Over/under line probability
    - Expected value calculation
    - Line shopping comparison
    """

    # Standard prop types
    PROP_TYPES = {
        'PTS': ['PTS'],
        'REB': ['REB'],
        'AST': ['AST'],
        'STL': ['STL'],
        'BLK': ['BLK'],
        'TOV': ['TOV'],
        'FG3M': ['FG3M'],
        'FTM': ['FTM'],
        'PTS_REB': ['PTS', 'REB'],
        'PTS_AST': ['PTS', 'AST'],
        'PTS_REB_AST': ['PTS', 'REB', 'AST'],
        'REB_AST': ['REB', 'AST'],
        'STL_BLK': ['STL', 'BLK'],
        'FANTASY': None,  # Special handling for fantasy points
    }

    def __init__(
        self,
        prop_type: str = 'PTS',
        model_name: str = 'player_props',
        use_distribution: bool = True,
        **model_params
    ):
        """
        Initialize player props predictor.

        Args:
            prop_type: Type of prop to predict (see PROP_TYPES)
            model_name: Name for model persistence
            use_distribution: Whether to model full distribution (more accurate)
        """
        self.prop_type = prop_type.upper()
        self.use_distribution = use_distribution
        self._distribution_params = {}
        super().__init__(model_name=model_name, **model_params)

    def _get_default_params(self) -> Dict:
        return {
            'max_iter': 300,
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_samples_leaf': 20,
            'random_state': 42,
        }

    def _build_model(self) -> Any:
        return HistGradientBoostingRegressor(**self.params)

    def _calculate_metrics(self, y_true, y_pred) -> Dict:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
        }

    def _get_target_column(self, df: pd.DataFrame) -> pd.Series:
        """Calculate target column based on prop type."""
        if self.prop_type == 'FANTASY':
            # DraftKings-style fantasy scoring
            return (
                df['PTS'] +
                df['REB'] * 1.25 +
                df['AST'] * 1.5 +
                df['STL'] * 2 +
                df['BLK'] * 2 -
                df['TOV'] * 0.5
            )

        if self.prop_type in self.PROP_TYPES:
            cols = self.PROP_TYPES[self.prop_type]
            return df[cols].sum(axis=1)

        # Try as single column
        if self.prop_type in df.columns:
            return df[self.prop_type]

        raise ValueError(f"Unknown prop type: {self.prop_type}")

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        validation_split: float = 0.2
    ) -> Dict:
        """Fit the props model."""
        df = df.copy()

        # Create target
        target = self._get_target_column(df)
        df['_prop_target'] = target

        # Engineer features
        engineered_df = self.feature_engineer.engineer_player_features(
            df, target_col='_prop_target'
        )

        # Get features
        if feature_cols is None:
            feature_cols = self.feature_engineer.get_feature_columns(
                engineered_df, '_prop_target'
            )
        self.feature_cols = feature_cols

        # Prepare data
        X, y = self.feature_engineer.prepare_training_data(
            engineered_df, '_prop_target', feature_cols
        )

        if len(X) < 100:
            raise ValueError(f"Insufficient data: {len(X)} samples")

        # Split
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        # Fit main model
        self.model = self._build_model()
        self.model.fit(X_train, y_train)

        # Fit distribution parameters for uncertainty
        if self.use_distribution:
            self._fit_distribution(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_val)
        val_metrics = self._calculate_metrics(y_val, y_pred)

        self.is_fitted = True
        self.target_col = '_prop_target'
        self.training_metadata = {
            'prop_type': self.prop_type,
            'n_samples': len(X),
            'n_features': len(feature_cols),
            'validation_metrics': val_metrics,
        }

        return self.training_metadata

    def _fit_distribution(self, X, y):
        """Fit player-specific distribution parameters."""
        # Compute residuals for variance estimation
        preds = self.model.predict(X)
        residuals = y.values - preds

        # Estimate variance model (heteroscedastic)
        # Variance tends to scale with prediction magnitude
        pred_buckets = pd.qcut(preds, q=10, duplicates='drop')
        self._distribution_params['bucket_std'] = {}

        for bucket in pred_buckets.unique():
            mask = pred_buckets == bucket
            if mask.sum() > 5:
                bucket_std = residuals[mask].std()
                self._distribution_params['bucket_std'][bucket] = bucket_std

        # Global fallback
        self._distribution_params['global_std'] = residuals.std()

    def predict_line_probability(
        self,
        df: pd.DataFrame,
        line: float,
        over: bool = True,
        engineer_features: bool = True
    ) -> np.ndarray:
        """
        Predict probability of over/under a line.

        Args:
            df: Player game data
            line: The betting line (e.g., 24.5 points)
            over: True for over probability, False for under
            engineer_features: Whether to engineer features

        Returns:
            Array of probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")

        if engineer_features:
            df = df.copy()
            df['_prop_target'] = self._get_target_column(df)
            df = self.feature_engineer.engineer_player_features(df, '_prop_target')

        X = df[self.feature_cols].fillna(df[self.feature_cols].median())
        predictions = self.model.predict(X)

        # Use distribution for probability
        if self.use_distribution:
            std = self._distribution_params.get('global_std', 5.0)
            # Normal distribution approximation
            z_scores = (line - predictions) / std
            probs = stats.norm.cdf(z_scores)

            if over:
                return 1 - probs
            return probs
        else:
            # Simple threshold
            if over:
                return (predictions > line).astype(float)
            return (predictions <= line).astype(float)

    def predict_player_prop(
        self,
        player_id: int,
        game_logs_df: pd.DataFrame,
        line: float,
        n_games_context: int = 20
    ) -> Dict:
        """
        Predict a specific player prop.

        Args:
            player_id: NBA player ID
            game_logs_df: DataFrame with player game logs
            line: The betting line
            n_games_context: Recent games to consider

        Returns:
            Dictionary with probabilities and analysis
        """
        player_df = game_logs_df[game_logs_df['PLAYER_ID'] == player_id].copy()

        if len(player_df) < 5:
            raise ValueError(f"Insufficient data for player {player_id}")

        player_df = player_df.sort_values('GAME_DATE').tail(n_games_context)

        # Get predictions for most recent context
        player_df['_prop_target'] = self._get_target_column(player_df)
        engineered = self.feature_engineer.engineer_player_features(
            player_df, '_prop_target'
        )

        last_row = engineered.tail(1)
        X = last_row[self.feature_cols].fillna(0)

        predicted_value = float(self.model.predict(X)[0])
        over_prob = float(self.predict_line_probability(
            last_row, line, over=True, engineer_features=False
        )[0])

        # Historical analysis
        actual_values = player_df['_prop_target'].values
        historical_over_rate = (actual_values > line).mean()
        recent_avg = actual_values[-5:].mean() if len(actual_values) >= 5 else actual_values.mean()

        return {
            'player_id': player_id,
            'prop_type': self.prop_type,
            'line': line,
            'predicted_value': predicted_value,
            'over_probability': over_prob,
            'under_probability': 1 - over_prob,
            'historical_over_rate': float(historical_over_rate),
            'recent_average': float(recent_avg),
            'games_analyzed': len(player_df),
            'edge_over': over_prob - 0.5,  # Edge vs 50/50
            'confidence': abs(over_prob - 0.5) * 2,  # 0-1 scale
        }

    def find_value_props(
        self,
        player_ids: List[int],
        game_logs_df: pd.DataFrame,
        lines: Dict[int, float],  # player_id -> line
        book_odds: Optional[Dict[int, Tuple[float, float]]] = None,  # player_id -> (over_odds, under_odds)
        min_edge: float = 0.03,
        n_games_context: int = 20
    ) -> List[Dict]:
        """
        Find value betting opportunities across multiple players.

        Args:
            player_ids: List of player IDs
            game_logs_df: Game logs data
            lines: Dictionary mapping player_id to betting line
            book_odds: Optional dict of (over_odds, under_odds) in decimal format
            min_edge: Minimum edge to consider (default 3%)

        Returns:
            List of value opportunities sorted by edge
        """
        opportunities = []

        for player_id in player_ids:
            if player_id not in lines:
                continue

            try:
                result = self.predict_player_prop(
                    player_id, game_logs_df, lines[player_id], n_games_context
                )

                # Calculate implied probability from odds
                if book_odds and player_id in book_odds:
                    over_odds, under_odds = book_odds[player_id]
                    implied_over = 1 / over_odds
                    implied_under = 1 / under_odds

                    # Edge calculation
                    over_edge = result['over_probability'] - implied_over
                    under_edge = result['under_probability'] - implied_under

                    if over_edge >= min_edge:
                        result['recommended_bet'] = 'OVER'
                        result['edge'] = over_edge
                        result['odds'] = over_odds
                        result['expected_value'] = over_edge * over_odds
                        opportunities.append(result)
                    elif under_edge >= min_edge:
                        result['recommended_bet'] = 'UNDER'
                        result['edge'] = under_edge
                        result['odds'] = under_odds
                        result['expected_value'] = under_edge * under_odds
                        opportunities.append(result)
                else:
                    # Without book odds, just flag high-confidence predictions
                    if result['confidence'] >= min_edge * 2:
                        if result['over_probability'] > 0.5:
                            result['recommended_bet'] = 'OVER'
                            result['edge'] = result['over_probability'] - 0.5
                        else:
                            result['recommended_bet'] = 'UNDER'
                            result['edge'] = result['under_probability'] - 0.5
                        opportunities.append(result)

            except Exception as e:
                warnings.warn(f"Error processing player {player_id}: {e}")
                continue

        # Sort by edge
        opportunities.sort(key=lambda x: x.get('edge', 0), reverse=True)

        return opportunities


class ComboPropsPredictor:
    """
    Predicts combination props (PTS+REB, PTS+AST, etc.).

    Handles correlation between stats for more accurate predictions.
    """

    def __init__(self, prop_types: List[str] = None):
        """
        Initialize combo props predictor.

        Args:
            prop_types: List of combo types to support
        """
        self.prop_types = prop_types or ['PTS_REB', 'PTS_AST', 'PTS_REB_AST']
        self.predictors = {}

    def fit(self, df: pd.DataFrame, **kwargs):
        """Fit predictors for all combo types."""
        for prop_type in self.prop_types:
            self.predictors[prop_type] = PlayerPropsPredictor(prop_type=prop_type)
            self.predictors[prop_type].fit(df, **kwargs)

    def predict_combo(
        self,
        player_id: int,
        game_logs_df: pd.DataFrame,
        prop_type: str,
        line: float,
        **kwargs
    ) -> Dict:
        """Predict a combo prop for a player."""
        if prop_type not in self.predictors:
            raise ValueError(f"Unknown prop type: {prop_type}")

        return self.predictors[prop_type].predict_player_prop(
            player_id, game_logs_df, line, **kwargs
        )


class AlternateLineAnalyzer:
    """
    Analyzes alternate lines for a player prop.

    Useful for finding the best value across different lines offered.
    """

    def __init__(self, props_predictor: PlayerPropsPredictor):
        self.predictor = props_predictor

    def analyze_alternate_lines(
        self,
        player_id: int,
        game_logs_df: pd.DataFrame,
        base_line: float,
        line_range: Tuple[float, float] = (-5, 5),
        step: float = 0.5,
        book_vig: float = 0.05,  # Assumed bookmaker vig
        n_games_context: int = 20
    ) -> pd.DataFrame:
        """
        Analyze value across alternate lines.

        Args:
            player_id: Player ID
            game_logs_df: Game logs
            base_line: Main line offered
            line_range: Range to analyze (relative to base)
            step: Line increments
            book_vig: Assumed bookmaker edge

        Returns:
            DataFrame with analysis for each line
        """
        results = []

        lines = np.arange(
            base_line + line_range[0],
            base_line + line_range[1] + step,
            step
        )

        for line in lines:
            try:
                pred = self.predictor.predict_player_prop(
                    player_id, game_logs_df, line, n_games_context
                )

                # Estimate fair odds (without vig)
                fair_over_odds = 1 / pred['over_probability'] if pred['over_probability'] > 0 else float('inf')
                fair_under_odds = 1 / pred['under_probability'] if pred['under_probability'] > 0 else float('inf')

                # Estimate book odds (with vig)
                book_over_odds = fair_over_odds * (1 - book_vig)
                book_under_odds = fair_under_odds * (1 - book_vig)

                results.append({
                    'line': line,
                    'over_prob': pred['over_probability'],
                    'under_prob': pred['under_probability'],
                    'fair_over_odds': fair_over_odds,
                    'fair_under_odds': fair_under_odds,
                    'est_book_over_odds': book_over_odds,
                    'est_book_under_odds': book_under_odds,
                    'over_edge_vs_base': pred['over_probability'] - 0.5,
                })
            except Exception:
                continue

        return pd.DataFrame(results)
