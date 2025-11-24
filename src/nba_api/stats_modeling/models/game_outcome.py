"""
Game Outcome Prediction Model

Predicts game outcomes (win/loss) using gradient boosting classification.
Can also predict win probability and point spread.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix
)
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

from nba_api.stats_modeling.models.base import NBABaseModel
from nba_api.stats_modeling.utils.features import FeatureEngineer

# Optional dependencies
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class GameOutcomePredictor(NBABaseModel):
    """
    Predicts game outcomes (win/loss) using gradient boosting classification.

    Features:
    - Team rolling statistics
    - Home court advantage
    - Win/loss streaks
    - Rest days differential
    - Head-to-head history

    Can predict:
    - Win probability (binary classification)
    - Point spread (regression)
    - Over/under totals (regression)
    """

    def __init__(
        self,
        backend: str = 'sklearn',
        prediction_type: str = 'win_probability',
        model_name: str = "game_outcome_predictor",
        **model_params
    ):
        """
        Initialize the game outcome predictor.

        Args:
            backend: ML backend ('lightgbm', 'xgboost', 'sklearn')
            prediction_type: 'win_probability', 'point_spread', or 'total_points'
            model_name: Name for the model
            **model_params: Model hyperparameters
        """
        self.backend = self._validate_backend(backend)
        self.prediction_type = prediction_type
        super().__init__(model_name=model_name, **model_params)
        self.label_encoder = LabelEncoder()

    def _validate_backend(self, backend: str) -> str:
        """Validate backend availability."""
        backend = backend.lower()
        if backend == 'lightgbm' and HAS_LIGHTGBM:
            return 'lightgbm'
        if backend == 'xgboost' and HAS_XGBOOST:
            return 'xgboost'
        return 'sklearn'

    def _is_classification(self) -> bool:
        """This is a classification model for win/loss."""
        return self.prediction_type == 'win_probability'

    def _get_default_params(self) -> Dict:
        """Return default hyperparameters."""
        if self.backend == 'lightgbm':
            return {
                'n_estimators': 300,
                'learning_rate': 0.05,
                'max_depth': 5,
                'num_leaves': 31,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1,
            }
        elif self.backend == 'xgboost':
            return {
                'n_estimators': 300,
                'learning_rate': 0.05,
                'max_depth': 5,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0,
            }
        else:
            return {
                'max_iter': 300,
                'learning_rate': 0.05,
                'max_depth': 5,
                'min_samples_leaf': 20,
                'l2_regularization': 0.1,
                'random_state': 42,
            }

    def _build_model(self) -> Any:
        """Build the classification or regression model."""
        is_classifier = self.prediction_type == 'win_probability'

        if self.backend == 'lightgbm':
            if is_classifier:
                return lgb.LGBMClassifier(**self.params)
            return lgb.LGBMRegressor(**self.params)
        elif self.backend == 'xgboost':
            if is_classifier:
                return xgb.XGBClassifier(**self.params)
            return xgb.XGBRegressor(**self.params)
        else:
            if is_classifier:
                return HistGradientBoostingClassifier(**self.params)
            return HistGradientBoostingRegressor(**self.params)

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate classification or regression metrics."""
        if self.prediction_type == 'win_probability':
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
                'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
            }
        else:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            return {
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred),
            }

    def engineer_game_features(self, team_logs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features specific to game outcome prediction.

        Args:
            team_logs_df: DataFrame with team game logs

        Returns:
            DataFrame with engineered features
        """
        df = team_logs_df.copy()

        # Ensure proper types
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df = df.sort_values(['TEAM_ID', 'GAME_DATE'])

        # Numeric conversions
        numeric_cols = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FGM', 'FGA',
                        'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB']

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Group by team
        team_groups = df.groupby('TEAM_ID')

        # Rolling averages for key stats
        windows = [3, 5, 10]
        for stat in ['PTS', 'REB', 'AST', 'FGM', 'FGA']:
            if stat not in df.columns:
                continue

            for window in windows:
                df[f'{stat}_avg_{window}'] = team_groups[stat].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )

        # Win/Loss encoding and streaks
        if 'WL' in df.columns:
            df['WIN'] = (df['WL'] == 'W').astype(int)

            # Win streak
            def calc_streak(series):
                streaks = []
                current = 0
                for val in series:
                    if pd.isna(val):
                        streaks.append(0)
                        continue
                    if val == 1:
                        current = max(0, current) + 1
                    else:
                        current = min(0, current) - 1
                    streaks.append(current)
                return streaks

            df['WIN_STREAK'] = team_groups['WIN'].transform(
                lambda x: pd.Series(calc_streak(x.shift(1).values), index=x.index)
            )

            # Rolling win rate
            for window in windows:
                df[f'WIN_RATE_{window}'] = team_groups['WIN'].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )

        # Home/Away indicator
        if 'MATCHUP' in df.columns:
            df['IS_HOME'] = (~df['MATCHUP'].str.contains('@', na=False)).astype(int)
        else:
            df['IS_HOME'] = 0

        # Days rest
        df['DAYS_REST'] = team_groups['GAME_DATE'].transform(
            lambda x: x.diff().dt.days.fillna(3)
        ).clip(upper=7)

        df['IS_BACK_TO_BACK'] = (df['DAYS_REST'] == 1).astype(int)

        # Offensive/Defensive efficiency proxies
        if all(c in df.columns for c in ['PTS', 'FGA', 'FTA', 'TOV']):
            # Possessions proxy
            df['POSS_PROXY'] = df['FGA'] + 0.44 * df['FTA'] + df['TOV']

            # Offensive rating proxy (points per possession)
            df['OFF_RATING_PROXY'] = df['PTS'] / (df['POSS_PROXY'] + 0.1) * 100

            # Rolling offensive rating
            for window in [5, 10]:
                df[f'OFF_RATING_{window}'] = team_groups['OFF_RATING_PROXY'].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )

        return df

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = 'WIN',
        feature_cols: Optional[List[str]] = None,
        validation_split: float = 0.2,
        engineer_features: bool = True
    ) -> Dict:
        """
        Fit the game outcome model.

        Args:
            df: DataFrame with team game logs
            target_col: Target column ('WIN', 'PTS', etc.)
            feature_cols: Feature columns (auto-detected if None)
            validation_split: Validation split ratio
            engineer_features: Whether to engineer features

        Returns:
            Training metadata
        """
        self.target_col = target_col

        # Engineer features for game prediction
        if engineer_features:
            df = self.engineer_game_features(df)

        # Ensure target exists
        if target_col not in df.columns:
            if 'WL' in df.columns and target_col == 'WIN':
                df['WIN'] = (df['WL'] == 'W').astype(int)
            else:
                raise ValueError(f"Target column '{target_col}' not found")

        # Get feature columns
        if feature_cols is None:
            feature_cols = self._get_game_feature_columns(df)

        self.feature_cols = feature_cols

        # Prepare data
        X = df[feature_cols].copy()
        y = df[target_col].copy()

        # Drop missing values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]

        if len(X) < 100:
            raise ValueError(f"Insufficient training data: {len(X)} samples")

        # Split for validation (time-series: no shuffle)
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, shuffle=False
        )

        # Build and train
        self.model = self._build_model()
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_val)
        val_metrics = self._calculate_metrics(y_val, y_pred)

        # Add probability metrics for classification
        if self.prediction_type == 'win_probability' and hasattr(self.model, 'predict_proba'):
            y_proba = self.model.predict_proba(X_val)[:, 1]
            val_metrics['roc_auc'] = roc_auc_score(y_val, y_proba)
            val_metrics['log_loss'] = log_loss(y_val, y_proba)

        self.is_fitted = True
        self.training_metadata = {
            'trained_at': pd.Timestamp.now().isoformat(),
            'n_samples': len(X),
            'n_features': len(feature_cols),
            'feature_cols': feature_cols,
            'target_col': target_col,
            'validation_metrics': val_metrics,
        }

        return self.training_metadata

    def _get_game_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get feature columns suitable for game prediction."""
        exclude = [
            'TEAM_ID', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL', 'WIN',
            'SEASON', 'SEASON_ID', 'VIDEO_AVAILABLE', 'TEAM_ABBREVIATION',
            'TEAM_NAME', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV',
            'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB',
            'PF', 'PLUS_MINUS', 'OFF_RATING_PROXY', 'POSS_PROXY',
        ]

        feature_cols = []
        for col in df.columns:
            if col in exclude:
                continue
            if df[col].dtype in [np.int64, np.float64, np.int32, np.float32]:
                feature_cols.append(col)

        return feature_cols

    def predict_proba(
        self,
        df: pd.DataFrame,
        engineer_features: bool = True
    ) -> np.ndarray:
        """
        Predict win probability.

        Args:
            df: DataFrame with team game logs
            engineer_features: Whether to engineer features

        Returns:
            Array of win probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")

        if self.prediction_type != 'win_probability':
            raise ValueError("predict_proba only available for win_probability model")

        if engineer_features:
            df = self.engineer_game_features(df)

        X = df[self.feature_cols].copy()
        X = X.fillna(X.median())

        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)[:, 1]
        else:
            # Fallback for models without predict_proba
            return self.model.predict(X)

    def predict_game(
        self,
        home_team_logs: pd.DataFrame,
        away_team_logs: pd.DataFrame,
        n_games_context: int = 10
    ) -> Dict:
        """
        Predict outcome for a specific game matchup.

        Args:
            home_team_logs: Recent game logs for home team
            away_team_logs: Recent game logs for away team
            n_games_context: Number of recent games to consider

        Returns:
            Dictionary with predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")

        # Get recent games
        home_recent = home_team_logs.sort_values('GAME_DATE').tail(n_games_context)
        away_recent = away_team_logs.sort_values('GAME_DATE').tail(n_games_context)

        # Engineer features
        home_feat = self.engineer_game_features(home_recent)
        away_feat = self.engineer_game_features(away_recent)

        # Get predictions from last entry (simulating next game)
        home_X = home_feat[self.feature_cols].tail(1).fillna(0)
        away_X = away_feat[self.feature_cols].tail(1).fillna(0)

        if self.prediction_type == 'win_probability' and hasattr(self.model, 'predict_proba'):
            home_win_prob = self.model.predict_proba(home_X)[:, 1][0]
            away_win_prob = self.model.predict_proba(away_X)[:, 1][0]
        else:
            home_win_prob = 0.5
            away_win_prob = 0.5

        # Normalize probabilities (since we predict separately)
        total = home_win_prob + (1 - away_win_prob)
        adjusted_home = home_win_prob / (total + 0.001)

        return {
            'home_win_probability': float(adjusted_home),
            'away_win_probability': float(1 - adjusted_home),
            'predicted_winner': 'home' if adjusted_home > 0.5 else 'away',
            'confidence': float(abs(adjusted_home - 0.5) * 2),
        }
