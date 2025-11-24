"""
Points Prediction Model

Predicts player points scored using gradient boosting regression.
Uses LightGBM for efficiency and XGBoost as alternative.
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

from nba_api.stats_modeling.models.base import NBABaseModel

# Try to import LightGBM and XGBoost (optional dependencies)
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


class PointsPredictor(NBABaseModel):
    """
    Predicts player points using gradient boosting.

    Features:
    - Rolling averages of past performance
    - Home/away adjustments
    - Rest days impact
    - Opponent strength (when available)

    Best Practices Applied:
    - Time-series validation (no future leakage)
    - Feature importance analysis
    - Confidence intervals
    - Robust to missing values
    """

    SUPPORTED_BACKENDS = ['lightgbm', 'xgboost', 'sklearn']

    def __init__(
        self,
        backend: str = 'sklearn',
        model_name: str = "points_predictor",
        **model_params
    ):
        """
        Initialize the points predictor.

        Args:
            backend: ML backend ('lightgbm', 'xgboost', or 'sklearn')
            model_name: Name for the model
            **model_params: Hyperparameters for the model
        """
        self.backend = self._validate_backend(backend)
        super().__init__(model_name=model_name, **model_params)

    def _validate_backend(self, backend: str) -> str:
        """Validate and select the best available backend."""
        backend = backend.lower()

        if backend == 'lightgbm':
            if HAS_LIGHTGBM:
                return 'lightgbm'
            print("LightGBM not installed, falling back to sklearn")
            return 'sklearn'

        if backend == 'xgboost':
            if HAS_XGBOOST:
                return 'xgboost'
            print("XGBoost not installed, falling back to sklearn")
            return 'sklearn'

        return 'sklearn'

    def _get_default_params(self) -> Dict:
        """Return default hyperparameters based on backend."""
        if self.backend == 'lightgbm':
            return {
                'n_estimators': 500,
                'learning_rate': 0.05,
                'max_depth': 6,
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
                'n_estimators': 500,
                'learning_rate': 0.05,
                'max_depth': 6,
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
                'max_iter': 500,
                'learning_rate': 0.05,
                'max_depth': 6,
                'min_samples_leaf': 20,
                'l2_regularization': 0.1,
                'random_state': 42,
            }

    def _build_model(self) -> Any:
        """Build the gradient boosting model."""
        if self.backend == 'lightgbm':
            return lgb.LGBMRegressor(**self.params)
        elif self.backend == 'xgboost':
            return xgb.XGBRegressor(**self.params)
        else:
            # Use sklearn's HistGradientBoosting (fast and robust)
            return HistGradientBoostingRegressor(**self.params)

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate regression metrics."""
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mean_actual': np.mean(y_true),
            'mean_predicted': np.mean(y_pred),
        }

    def predict_player(
        self,
        player_id: int,
        game_logs_df: pd.DataFrame,
        n_games_context: int = 20
    ) -> Dict:
        """
        Predict points for a specific player based on recent performance.

        Args:
            player_id: NBA player ID
            game_logs_df: DataFrame with player's game logs
            n_games_context: Number of recent games for context

        Returns:
            Dictionary with prediction and details
        """
        player_df = game_logs_df[game_logs_df['PLAYER_ID'] == player_id].copy()

        if len(player_df) < 5:
            raise ValueError(f"Insufficient data for player {player_id}")

        # Get last n_games for context
        player_df = player_df.sort_values('GAME_DATE').tail(n_games_context)

        # Make prediction on most recent entry (simulating next game)
        pred, lower, upper = self.predict_with_confidence(
            player_df.tail(1),
            engineer_features=True
        )

        # Get recent averages for context
        recent_avg = player_df['PTS'].mean() if 'PTS' in player_df.columns else None

        return {
            'player_id': player_id,
            'predicted_points': float(pred[0]),
            'confidence_lower': float(lower[0]),
            'confidence_upper': float(upper[0]),
            'recent_average': recent_avg,
            'games_analyzed': len(player_df),
        }


class PointsPredictorEnsemble:
    """
    Ensemble model combining multiple points predictors.

    Averages predictions from multiple models for more robust estimates.
    """

    def __init__(self, models: Optional[list] = None):
        """
        Initialize the ensemble.

        Args:
            models: List of PointsPredictor instances
        """
        if models is None:
            # Create default ensemble
            self.models = [
                PointsPredictor(backend='sklearn', model_name='points_sklearn'),
            ]

            # Add optional backends if available
            if HAS_LIGHTGBM:
                self.models.append(
                    PointsPredictor(backend='lightgbm', model_name='points_lgbm')
                )
            if HAS_XGBOOST:
                self.models.append(
                    PointsPredictor(backend='xgboost', model_name='points_xgb')
                )
        else:
            self.models = models

        self.is_fitted = False

    def fit(self, df: pd.DataFrame, target_col: str = 'PTS', **kwargs):
        """Fit all models in the ensemble."""
        for model in self.models:
            print(f"\nTraining {model.model_name}...")
            model.fit(df, target_col, **kwargs)

        self.is_fitted = True

    def predict(self, df: pd.DataFrame, **kwargs) -> np.ndarray:
        """Average predictions from all models."""
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted first")

        predictions = []
        for model in self.models:
            pred = model.predict(df, **kwargs)
            predictions.append(pred)

        # Average predictions
        return np.mean(predictions, axis=0)

    def predict_with_uncertainty(self, df: pd.DataFrame, **kwargs) -> Dict:
        """
        Get predictions with uncertainty from ensemble disagreement.

        Returns:
            Dictionary with mean prediction and model disagreement
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted first")

        predictions = []
        for model in self.models:
            pred = model.predict(df, **kwargs)
            predictions.append(pred)

        predictions = np.array(predictions)

        return {
            'mean': np.mean(predictions, axis=0),
            'std': np.std(predictions, axis=0),
            'min': np.min(predictions, axis=0),
            'max': np.max(predictions, axis=0),
        }
