"""
Individual Statistics Prediction Models

Predicts various player statistics:
- Three-pointers made (FG3M)
- Free throws made (FTM)
- Blocks (BLK)
- Field goals made (FGM)

Uses multi-target learning for efficiency and shared feature engineering.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

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


class IndividualStatsPredictor(NBABaseModel):
    """
    Multi-target predictor for individual player statistics.

    Predicts multiple stats simultaneously:
    - FG3M: Three-pointers made
    - FTM: Free throws made
    - BLK: Blocks
    - FGM: Field goals made

    Benefits of multi-target approach:
    - Shared feature engineering
    - Captures correlations between stats
    - More efficient training
    """

    # Default stats to predict
    DEFAULT_TARGETS = ['FG3M', 'FTM', 'BLK', 'FGM']

    def __init__(
        self,
        targets: Optional[List[str]] = None,
        backend: str = 'sklearn',
        model_name: str = "individual_stats_predictor",
        **model_params
    ):
        """
        Initialize the multi-target predictor.

        Args:
            targets: List of target statistics to predict
            backend: ML backend ('lightgbm', 'xgboost', 'sklearn')
            model_name: Name for the model
            **model_params: Model hyperparameters
        """
        self.targets = targets or self.DEFAULT_TARGETS
        self.backend = self._validate_backend(backend)
        self._individual_models = {}
        super().__init__(model_name=model_name, **model_params)

    def _validate_backend(self, backend: str) -> str:
        """Validate backend availability."""
        backend = backend.lower()
        if backend == 'lightgbm' and HAS_LIGHTGBM:
            return 'lightgbm'
        if backend == 'xgboost' and HAS_XGBOOST:
            return 'xgboost'
        return 'sklearn'

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
        """Build the multi-output regression model."""
        if self.backend == 'lightgbm':
            base_model = lgb.LGBMRegressor(**self.params)
        elif self.backend == 'xgboost':
            base_model = xgb.XGBRegressor(**self.params)
        else:
            base_model = HistGradientBoostingRegressor(**self.params)

        return MultiOutputRegressor(base_model)

    def _build_single_model(self) -> Any:
        """Build a single-output model for individual training."""
        if self.backend == 'lightgbm':
            return lgb.LGBMRegressor(**self.params)
        elif self.backend == 'xgboost':
            return xgb.XGBRegressor(**self.params)
        else:
            return HistGradientBoostingRegressor(**self.params)

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate metrics for multi-output regression."""
        metrics = {}

        # If y_true/y_pred are 2D (multi-output)
        if y_true.ndim == 2:
            for i, target in enumerate(self.targets):
                metrics[f'{target}_rmse'] = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
                metrics[f'{target}_mae'] = mean_absolute_error(y_true[:, i], y_pred[:, i])
                metrics[f'{target}_r2'] = r2_score(y_true[:, i], y_pred[:, i])

            # Overall metrics
            metrics['overall_rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics['overall_mae'] = mean_absolute_error(y_true, y_pred)
        else:
            metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)

        return metrics

    def engineer_stat_features(
        self,
        df: pd.DataFrame,
        target_stat: str
    ) -> pd.DataFrame:
        """
        Engineer features specific to a target statistic.

        Different stats may benefit from different features:
        - FG3M: Three-point attempt rate, shooting percentage trends
        - FTM: Free throw attempt rate, fouling context
        - BLK: Playing time, position, opponent size
        - FGM: Overall shooting volume and efficiency
        """
        df = df.copy()

        # Common feature engineering
        df = self.feature_engineer.engineer_player_features(df, target_stat)

        # Stat-specific features
        if target_stat == 'FG3M':
            df = self._add_three_point_features(df)
        elif target_stat == 'FTM':
            df = self._add_free_throw_features(df)
        elif target_stat == 'BLK':
            df = self._add_blocks_features(df)
        elif target_stat == 'FGM':
            df = self._add_field_goal_features(df)

        return df

    def _add_three_point_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add three-point specific features."""
        if 'FG3A' in df.columns and 'FGA' in df.columns:
            # Three-point attempt rate
            df['FG3_RATE'] = df['FG3A'] / (df['FGA'] + 0.1)

            # Rolling three-point rate
            df['FG3_RATE_rolling'] = df.groupby('PLAYER_ID')['FG3_RATE'].transform(
                lambda x: x.shift(1).rolling(10, min_periods=3).mean()
            )

        return df

    def _add_free_throw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add free throw specific features."""
        if 'FTA' in df.columns and 'MIN' in df.columns:
            # Free throw attempts per minute
            df['FTA_PER_MIN'] = df['FTA'] / (pd.to_numeric(df['MIN'], errors='coerce') + 0.1)

            # Rolling FTA rate
            df['FTA_RATE_rolling'] = df.groupby('PLAYER_ID')['FTA_PER_MIN'].transform(
                lambda x: x.shift(1).rolling(10, min_periods=3).mean()
            )

        return df

    def _add_blocks_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add blocks-specific features."""
        if 'BLK' in df.columns and 'MIN' in df.columns:
            # Blocks per minute
            df['BLK_PER_MIN'] = df['BLK'] / (pd.to_numeric(df['MIN'], errors='coerce') + 0.1)

            # Rolling blocks rate
            df['BLK_RATE_rolling'] = df.groupby('PLAYER_ID')['BLK_PER_MIN'].transform(
                lambda x: x.shift(1).rolling(10, min_periods=3).mean()
            )

        return df

    def _add_field_goal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add field goal specific features."""
        if 'FGA' in df.columns and 'MIN' in df.columns:
            # Shot attempts per minute (usage proxy)
            df['FGA_PER_MIN'] = df['FGA'] / (pd.to_numeric(df['MIN'], errors='coerce') + 0.1)

            # Rolling FGA rate
            df['FGA_RATE_rolling'] = df.groupby('PLAYER_ID')['FGA_PER_MIN'].transform(
                lambda x: x.shift(1).rolling(10, min_periods=3).mean()
            )

        return df

    def fit(
        self,
        df: pd.DataFrame,
        targets: Optional[List[str]] = None,
        feature_cols: Optional[List[str]] = None,
        validation_split: float = 0.2,
        strategy: str = 'individual'
    ) -> Dict:
        """
        Fit the multi-target model.

        Args:
            df: DataFrame with player game logs
            targets: Target columns (uses default if None)
            feature_cols: Feature columns (auto-detected if None)
            validation_split: Validation split ratio
            strategy: 'individual' (separate models) or 'multi' (single multi-output)

        Returns:
            Training metadata
        """
        targets = targets or self.targets
        self.targets = targets

        # Engineer features once
        engineered_df = self.feature_engineer.engineer_player_features(df, targets[0])

        # Get feature columns
        if feature_cols is None:
            feature_cols = self.feature_engineer.get_feature_columns(engineered_df, targets[0])
            # Also exclude other targets
            feature_cols = [c for c in feature_cols if c not in targets]

        self.feature_cols = feature_cols

        # Prepare data
        X = engineered_df[feature_cols].copy()
        y = engineered_df[targets].copy()

        # Drop missing
        mask = ~(X.isna().any(axis=1) | y.isna().any(axis=1))
        X = X[mask]
        y = y[mask]

        if len(X) < 100:
            raise ValueError(f"Insufficient training data: {len(X)} samples")

        # Split
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, shuffle=False
        )

        all_metrics = {}

        if strategy == 'multi':
            # Single multi-output model
            self.model = self._build_model()
            self.model.fit(X_train, y_train)

            y_pred = self.model.predict(X_val)
            all_metrics = self._calculate_metrics(y_val.values, y_pred)

        else:
            # Individual models for each target
            self._individual_models = {}
            for target in targets:
                model = self._build_single_model()
                model.fit(X_train, y_train[target])
                self._individual_models[target] = model

                y_pred = model.predict(X_val)
                metrics = self._calculate_metrics(y_val[target].values, y_pred)
                for k, v in metrics.items():
                    all_metrics[f'{target}_{k}'] = v

            # Set model to first one for compatibility
            self.model = self._individual_models[targets[0]]

        self.is_fitted = True
        self.training_metadata = {
            'trained_at': pd.Timestamp.now().isoformat(),
            'n_samples': len(X),
            'n_features': len(feature_cols),
            'targets': targets,
            'feature_cols': feature_cols,
            'strategy': strategy,
            'validation_metrics': all_metrics,
        }

        return self.training_metadata

    def predict(
        self,
        df: pd.DataFrame,
        engineer_features: bool = True
    ) -> pd.DataFrame:
        """
        Predict all target statistics.

        Returns:
            DataFrame with predictions for each target
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")

        if engineer_features:
            df = self.feature_engineer.engineer_player_features(df, self.targets[0])

        X = df[self.feature_cols].copy()
        X = X.fillna(X.median())

        predictions = pd.DataFrame(index=X.index)

        if hasattr(self, '_individual_models') and self._individual_models:
            # Individual models
            for target, model in self._individual_models.items():
                predictions[f'{target}_pred'] = model.predict(X)
        else:
            # Multi-output model
            preds = self.model.predict(X)
            for i, target in enumerate(self.targets):
                predictions[f'{target}_pred'] = preds[:, i]

        return predictions

    def predict_player_stats(
        self,
        player_id: int,
        game_logs_df: pd.DataFrame,
        n_games_context: int = 20
    ) -> Dict:
        """
        Predict all stats for a specific player.

        Args:
            player_id: NBA player ID
            game_logs_df: DataFrame with player's game logs
            n_games_context: Number of recent games for context

        Returns:
            Dictionary with predictions for all target stats
        """
        player_df = game_logs_df[game_logs_df['PLAYER_ID'] == player_id].copy()

        if len(player_df) < 5:
            raise ValueError(f"Insufficient data for player {player_id}")

        player_df = player_df.sort_values('GAME_DATE').tail(n_games_context)

        # Get predictions
        predictions = self.predict(player_df.tail(1), engineer_features=True)

        # Get recent averages
        result = {
            'player_id': player_id,
            'games_analyzed': len(player_df),
            'predictions': {},
            'recent_averages': {},
        }

        for target in self.targets:
            pred_col = f'{target}_pred'
            if pred_col in predictions.columns:
                result['predictions'][target] = float(predictions[pred_col].iloc[0])

            if target in player_df.columns:
                result['recent_averages'][target] = float(player_df[target].mean())

        return result


class ThreePointPredictor(IndividualStatsPredictor):
    """Specialized predictor for three-pointers made."""

    def __init__(self, **kwargs):
        super().__init__(targets=['FG3M'], model_name='three_point_predictor', **kwargs)


class FreeThrowPredictor(IndividualStatsPredictor):
    """Specialized predictor for free throws made."""

    def __init__(self, **kwargs):
        super().__init__(targets=['FTM'], model_name='free_throw_predictor', **kwargs)


class BlocksPredictor(IndividualStatsPredictor):
    """Specialized predictor for blocks."""

    def __init__(self, **kwargs):
        super().__init__(targets=['BLK'], model_name='blocks_predictor', **kwargs)


class FieldGoalPredictor(IndividualStatsPredictor):
    """Specialized predictor for field goals made."""

    def __init__(self, **kwargs):
        super().__init__(targets=['FGM'], model_name='field_goal_predictor', **kwargs)
