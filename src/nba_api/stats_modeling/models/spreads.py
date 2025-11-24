"""
Spread and Totals Prediction Models

Professional-grade models for predicting:
- Point spreads (margin of victory)
- Game totals (over/under)
- First half/quarter lines
- Live game projections
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
import warnings

from nba_api.stats_modeling.models.base import NBABaseModel
from nba_api.stats_modeling.models.ensemble import StackedEnsembleRegressor, BayesianModelAveraging


class SpreadPredictor(NBABaseModel):
    """
    Predicts point spreads (margin of victory) for NBA games.

    Features:
    - Team strength differential
    - Home court advantage quantification
    - Rest advantage/disadvantage
    - Travel fatigue modeling
    - Recent form momentum
    """

    # Average NBA home court advantage (points)
    HOME_COURT_ADVANTAGE = 3.0

    def __init__(
        self,
        use_ensemble: bool = True,
        model_name: str = 'spread_predictor',
        **model_params
    ):
        self.use_ensemble = use_ensemble
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
        if self.use_ensemble:
            return StackedEnsembleRegressor(
                use_neural_net=True,
                use_advanced_boosting=True,
                meta_learner='ridge'
            )
        return HistGradientBoostingRegressor(**self.params)

    def _calculate_metrics(self, y_true, y_pred) -> Dict:
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        errors = y_true - y_pred

        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mean_error': float(np.mean(errors)),
            'std_error': float(np.std(errors)),
            'cover_rate_vs_zero': float((errors * y_pred > 0).mean()),
        }

    def engineer_spread_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features specific to spread prediction."""
        df = df.copy()
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df = df.sort_values(['TEAM_ID', 'GAME_DATE'])

        # Numeric conversions
        numeric_cols = ['PTS', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA',
                        'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF']

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        groups = df.groupby('TEAM_ID')

        # Calculate margin (positive = win)
        if 'PLUS_MINUS' in df.columns:
            df['MARGIN'] = pd.to_numeric(df['PLUS_MINUS'], errors='coerce')
        elif 'PTS' in df.columns:
            # Need to calculate from opponent points
            df['MARGIN'] = 0  # Placeholder

        # Rolling offensive/defensive stats
        windows = [3, 5, 10]

        for window in windows:
            # Points scored (offense)
            df[f'PTS_avg_{window}'] = groups['PTS'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

            # Points allowed (defense) - using margin
            if 'MARGIN' in df.columns:
                df[f'MARGIN_avg_{window}'] = groups['MARGIN'].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )

            # Win percentage
            if 'WL' in df.columns:
                df['WIN'] = (df['WL'] == 'W').astype(int)
                df[f'WIN_PCT_{window}'] = groups['WIN'].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )

        # Offensive efficiency proxies
        if all(c in df.columns for c in ['PTS', 'FGA', 'FTA', 'TOV']):
            df['POSS_EST'] = df['FGA'] + 0.44 * df['FTA'] + df['TOV']
            df['OFF_RTG'] = df['PTS'] / (df['POSS_EST'] + 0.1) * 100

            for window in [5, 10]:
                df[f'OFF_RTG_{window}'] = groups['OFF_RTG'].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )

        # Pace (possessions per game)
        if 'POSS_EST' in df.columns:
            for window in [5, 10]:
                df[f'PACE_{window}'] = groups['POSS_EST'].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )

        # Rest days
        df['DAYS_REST'] = groups['GAME_DATE'].transform(
            lambda x: x.diff().dt.days.fillna(3)
        ).clip(upper=7)

        df['IS_BACK_TO_BACK'] = (df['DAYS_REST'] == 1).astype(int)
        df['IS_WELL_RESTED'] = (df['DAYS_REST'] >= 3).astype(int)

        # Home/Away
        if 'MATCHUP' in df.columns:
            df['IS_HOME'] = (~df['MATCHUP'].str.contains('@', na=False)).astype(int)
        else:
            df['IS_HOME'] = 0

        # Win/Loss streaks
        if 'WIN' in df.columns:
            df['WIN_STREAK'] = groups['WIN'].transform(
                lambda x: self._calculate_streak(x.shift(1).values)
            )

        return df

    def _calculate_streak(self, win_values) -> np.ndarray:
        """Calculate current win/loss streak."""
        streaks = np.zeros(len(win_values))
        current = 0

        for i, win in enumerate(win_values):
            if pd.isna(win):
                current = 0
            elif win == 1:
                current = max(0, current) + 1
            else:
                current = min(0, current) - 1
            streaks[i] = current

        return streaks

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = 'MARGIN',
        feature_cols: Optional[List[str]] = None,
        validation_split: float = 0.2
    ) -> Dict:
        """Fit the spread prediction model."""
        self.target_col = target_col

        # Engineer features
        engineered_df = self.engineer_spread_features(df)

        # Create target if not exists
        if target_col not in engineered_df.columns and 'PLUS_MINUS' in engineered_df.columns:
            engineered_df[target_col] = pd.to_numeric(
                engineered_df['PLUS_MINUS'], errors='coerce'
            )

        # Auto-select features
        if feature_cols is None:
            feature_cols = self._get_spread_features(engineered_df)

        self.feature_cols = feature_cols

        # Prepare data
        X = engineered_df[feature_cols].copy()
        y = engineered_df[target_col].copy()

        # Drop NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]

        if len(X) < 100:
            raise ValueError(f"Insufficient data: {len(X)} samples")

        # Time-series split
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        # Fit model
        self.model = self._build_model()
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_val)
        val_metrics = self._calculate_metrics(y_val.values, y_pred)

        self.is_fitted = True
        self.training_metadata = {
            'n_samples': len(X),
            'n_features': len(feature_cols),
            'validation_metrics': val_metrics,
        }

        return self.training_metadata

    def _get_spread_features(self, df: pd.DataFrame) -> List[str]:
        """Get feature columns for spread prediction."""
        exclude = ['TEAM_ID', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL', 'WIN',
                   'MARGIN', 'PLUS_MINUS', 'SEASON', 'PTS', 'VIDEO_AVAILABLE']

        feature_cols = []
        for col in df.columns:
            if col in exclude:
                continue
            if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                feature_cols.append(col)

        return feature_cols

    def predict_spread(
        self,
        home_team_logs: pd.DataFrame,
        away_team_logs: pd.DataFrame,
        n_games_context: int = 10
    ) -> Dict:
        """
        Predict spread for a specific matchup.

        Args:
            home_team_logs: Recent logs for home team
            away_team_logs: Recent logs for away team
            n_games_context: Recent games to consider

        Returns:
            Dictionary with spread prediction
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")

        # Process both teams
        home_df = home_team_logs.sort_values('GAME_DATE').tail(n_games_context)
        away_df = away_team_logs.sort_values('GAME_DATE').tail(n_games_context)

        home_feat = self.engineer_spread_features(home_df)
        away_feat = self.engineer_spread_features(away_df)

        # Get most recent features
        home_X = home_feat[self.feature_cols].tail(1).fillna(0)
        away_X = away_feat[self.feature_cols].tail(1).fillna(0)

        # Predict each team's expected margin
        home_margin = float(self.model.predict(home_X)[0])
        away_margin = float(self.model.predict(away_X)[0])

        # Combined prediction with home court
        # Home team margin = home_strength - away_strength + HCA
        predicted_spread = (home_margin - away_margin) / 2 + self.HOME_COURT_ADVANTAGE

        # Uncertainty estimation
        std_error = self.training_metadata.get(
            'validation_metrics', {}
        ).get('std_error', 10.0)

        return {
            'predicted_spread': predicted_spread,
            'home_favored': predicted_spread > 0,
            'spread_magnitude': abs(predicted_spread),
            'home_margin_strength': home_margin,
            'away_margin_strength': away_margin,
            'confidence_interval_68': (predicted_spread - std_error,
                                        predicted_spread + std_error),
            'confidence_interval_95': (predicted_spread - 2*std_error,
                                        predicted_spread + 2*std_error),
        }

    def predict_spread_probability(
        self,
        home_team_logs: pd.DataFrame,
        away_team_logs: pd.DataFrame,
        line: float,
        n_games_context: int = 10
    ) -> Dict:
        """
        Predict probability of covering a spread.

        Args:
            home_team_logs: Home team data
            away_team_logs: Away team data
            line: The spread line (negative = home favored)
            n_games_context: Games to consider

        Returns:
            Cover probabilities
        """
        pred = self.predict_spread(home_team_logs, away_team_logs, n_games_context)

        std_error = self.training_metadata.get(
            'validation_metrics', {}
        ).get('std_error', 10.0)

        # Home team covers if actual margin > line
        # P(margin > line) = P(Z > (line - pred) / std)
        z_score = (line - pred['predicted_spread']) / std_error
        home_cover_prob = 1 - stats.norm.cdf(z_score)

        return {
            **pred,
            'line': line,
            'home_cover_probability': float(home_cover_prob),
            'away_cover_probability': float(1 - home_cover_prob),
            'push_probability': float(stats.norm.pdf(z_score) * 0.5),  # Approximate
        }


class TotalsPredictor(NBABaseModel):
    """
    Predicts game totals (combined score) for over/under betting.

    Factors:
    - Team pace (possessions per game)
    - Offensive/defensive efficiency
    - Rest impact on pace
    - Historical matchup pace
    """

    def __init__(
        self,
        use_ensemble: bool = True,
        model_name: str = 'totals_predictor',
        **model_params
    ):
        self.use_ensemble = use_ensemble
        super().__init__(model_name=model_name, **model_params)

    def _get_default_params(self) -> Dict:
        return {
            'max_iter': 300,
            'learning_rate': 0.05,
            'max_depth': 6,
            'random_state': 42,
        }

    def _build_model(self) -> Any:
        if self.use_ensemble:
            return BayesianModelAveraging(
                task='regression',
                n_bootstrap=50
            )
        return HistGradientBoostingRegressor(**self.params)

    def _calculate_metrics(self, y_true, y_pred) -> Dict:
        from sklearn.metrics import mean_squared_error, mean_absolute_error

        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mean_total': float(np.mean(y_true)),
            'predicted_mean': float(np.mean(y_pred)),
        }

    def engineer_totals_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for totals prediction."""
        df = df.copy()
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df = df.sort_values(['TEAM_ID', 'GAME_DATE'])

        numeric_cols = ['PTS', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'TOV']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        groups = df.groupby('TEAM_ID')

        # Pace estimation
        if all(c in df.columns for c in ['FGA', 'FTA', 'TOV']):
            df['POSS'] = df['FGA'] + 0.44 * df['FTA'] + df['TOV']

            for window in [5, 10, 20]:
                df[f'PACE_{window}'] = groups['POSS'].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )

        # Scoring averages
        if 'PTS' in df.columns:
            for window in [5, 10, 20]:
                df[f'PTS_avg_{window}'] = groups['PTS'].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )

                df[f'PTS_std_{window}'] = groups['PTS'].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).std()
                )

        # Three-point shooting (affects totals)
        if all(c in df.columns for c in ['FG3M', 'FG3A']):
            df['FG3_RATE'] = df['FG3M'] / (df['FG3A'] + 0.1)
            df['FG3A_PER_GAME'] = df['FG3A']

            for window in [5, 10]:
                df[f'FG3_RATE_{window}'] = groups['FG3_RATE'].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )
                df[f'FG3A_{window}'] = groups['FG3A_PER_GAME'].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )

        # Rest affects pace
        df['DAYS_REST'] = groups['GAME_DATE'].transform(
            lambda x: x.diff().dt.days.fillna(3)
        ).clip(upper=7)

        df['IS_BACK_TO_BACK'] = (df['DAYS_REST'] == 1).astype(int)

        return df

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = 'PTS',  # Team points (will double for total)
        feature_cols: Optional[List[str]] = None,
        validation_split: float = 0.2
    ) -> Dict:
        """Fit totals prediction model."""
        self.target_col = target_col

        engineered_df = self.engineer_totals_features(df)

        if feature_cols is None:
            feature_cols = self._get_totals_features(engineered_df)

        self.feature_cols = feature_cols

        X = engineered_df[feature_cols].copy()
        y = engineered_df[target_col].copy()

        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]

        if len(X) < 100:
            raise ValueError(f"Insufficient data: {len(X)}")

        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        self.model = self._build_model()
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_val)
        val_metrics = self._calculate_metrics(y_val.values, y_pred)

        self.is_fitted = True
        self.training_metadata = {
            'n_samples': len(X),
            'n_features': len(feature_cols),
            'validation_metrics': val_metrics,
        }

        return self.training_metadata

    def _get_totals_features(self, df: pd.DataFrame) -> List[str]:
        """Get features for totals prediction."""
        exclude = ['TEAM_ID', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL', 'WIN',
                   'MARGIN', 'PLUS_MINUS', 'SEASON', 'PTS', 'VIDEO_AVAILABLE',
                   'FG3_RATE', 'POSS', 'FG3A_PER_GAME']

        return [col for col in df.columns
                if col not in exclude and
                df[col].dtype in [np.float64, np.int64, np.float32, np.int32]]

    def predict_total(
        self,
        home_team_logs: pd.DataFrame,
        away_team_logs: pd.DataFrame,
        n_games_context: int = 10
    ) -> Dict:
        """
        Predict game total for a matchup.

        Returns predicted combined score.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")

        home_df = home_team_logs.sort_values('GAME_DATE').tail(n_games_context)
        away_df = away_team_logs.sort_values('GAME_DATE').tail(n_games_context)

        home_feat = self.engineer_totals_features(home_df)
        away_feat = self.engineer_totals_features(away_df)

        home_X = home_feat[self.feature_cols].tail(1).fillna(0)
        away_X = away_feat[self.feature_cols].tail(1).fillna(0)

        home_pts = float(self.model.predict(home_X)[0])
        away_pts = float(self.model.predict(away_X)[0])

        predicted_total = home_pts + away_pts

        # Uncertainty
        rmse = self.training_metadata.get('validation_metrics', {}).get('rmse', 8.0)
        total_std = rmse * np.sqrt(2)  # Combined variance

        return {
            'predicted_total': predicted_total,
            'home_points': home_pts,
            'away_points': away_pts,
            'std_error': total_std,
            'confidence_interval_68': (predicted_total - total_std,
                                        predicted_total + total_std),
            'confidence_interval_95': (predicted_total - 2*total_std,
                                        predicted_total + 2*total_std),
        }

    def predict_total_probability(
        self,
        home_team_logs: pd.DataFrame,
        away_team_logs: pd.DataFrame,
        line: float,
        n_games_context: int = 10
    ) -> Dict:
        """
        Predict probability of over/under.

        Args:
            home_team_logs: Home team data
            away_team_logs: Away team data
            line: The total line (e.g., 220.5)

        Returns:
            Over/under probabilities
        """
        pred = self.predict_total(home_team_logs, away_team_logs, n_games_context)

        z_score = (line - pred['predicted_total']) / pred['std_error']
        over_prob = 1 - stats.norm.cdf(z_score)

        return {
            **pred,
            'line': line,
            'over_probability': float(over_prob),
            'under_probability': float(1 - over_prob),
            'edge_over': float(over_prob - 0.5),
            'edge_under': float(0.5 - over_prob),
        }
