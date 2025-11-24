"""
Base Model Class for NBA Statistics Prediction

Provides common functionality for all prediction models including:
- Feature engineering integration
- Model persistence
- Cross-validation
- Prediction with confidence intervals
"""

import os
import pickle
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from nba_api.stats_modeling.utils.features import FeatureEngineer
from nba_api.stats_modeling.utils.validation import CrossValidator


class NBABaseModel(ABC):
    """
    Abstract base class for NBA prediction models.

    All models should inherit from this class and implement:
    - _build_model(): Create the underlying ML model
    - _get_default_params(): Return default hyperparameters
    """

    def __init__(
        self,
        model_name: str = "nba_model",
        model_dir: str = "./models",
        feature_engineer: Optional[FeatureEngineer] = None,
        **model_params
    ):
        """
        Initialize the base model.

        Args:
            model_name: Name for saving/loading the model
            model_dir: Directory for model persistence
            feature_engineer: Feature engineering instance
            **model_params: Model-specific hyperparameters
        """
        self.model_name = model_name
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.feature_engineer = feature_engineer or FeatureEngineer()

        # Merge default params with provided params
        self.params = self._get_default_params()
        self.params.update(model_params)

        self.model = None
        self.feature_cols = None
        self.target_col = None
        self.is_fitted = False
        self.training_metadata = {}

    @abstractmethod
    def _build_model(self) -> Any:
        """Build and return the underlying ML model."""
        pass

    @abstractmethod
    def _get_default_params(self) -> Dict:
        """Return default hyperparameters for the model."""
        pass

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        validation_split: float = 0.2,
        engineer_features: bool = True
    ) -> Dict:
        """
        Fit the model on training data.

        Args:
            df: DataFrame with game/player data
            target_col: Column to predict
            feature_cols: Feature columns (auto-detected if None)
            validation_split: Fraction for validation
            engineer_features: Whether to apply feature engineering

        Returns:
            Dictionary with training results
        """
        self.target_col = target_col

        # Feature engineering
        if engineer_features:
            df = self.feature_engineer.engineer_player_features(df, target_col)

        # Get feature columns
        if feature_cols is None:
            feature_cols = self.feature_engineer.get_feature_columns(df, target_col)

        self.feature_cols = feature_cols

        # Prepare training data
        X, y = self.feature_engineer.prepare_training_data(
            df, target_col, feature_cols
        )

        if len(X) < 100:
            raise ValueError(f"Insufficient training data: {len(X)} samples")

        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, shuffle=False  # Time-series: no shuffle
        )

        # Build and train model
        self.model = self._build_model()
        self.model.fit(X_train, y_train)

        # Evaluate on validation set
        y_pred = self.model.predict(X_val)
        val_metrics = self._calculate_metrics(y_val, y_pred)

        self.is_fitted = True
        self.training_metadata = {
            'trained_at': datetime.now().isoformat(),
            'n_samples': len(X),
            'n_features': len(feature_cols),
            'feature_cols': feature_cols,
            'target_col': target_col,
            'validation_metrics': val_metrics,
        }

        return self.training_metadata

    def predict(
        self,
        df: pd.DataFrame,
        engineer_features: bool = True
    ) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            df: DataFrame with game/player data
            engineer_features: Whether to apply feature engineering

        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")

        if engineer_features:
            df = self.feature_engineer.engineer_player_features(df, self.target_col)

        X = df[self.feature_cols].copy()

        # Handle missing values
        X = X.fillna(X.median())

        return self.model.predict(X)

    def predict_with_confidence(
        self,
        df: pd.DataFrame,
        confidence_level: float = 0.9,
        engineer_features: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with confidence intervals.

        Args:
            df: DataFrame with game/player data
            confidence_level: Confidence level for intervals
            engineer_features: Whether to apply feature engineering

        Returns:
            Tuple of (predictions, lower_bound, upper_bound)
        """
        predictions = self.predict(df, engineer_features)

        # Estimate prediction intervals based on validation error
        if 'validation_metrics' in self.training_metadata:
            rmse = self.training_metadata['validation_metrics'].get('rmse', 5.0)
        else:
            rmse = 5.0  # Default estimate

        # Use normal distribution approximation
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence_level) / 2)

        margin = z_score * rmse
        lower = predictions - margin
        upper = predictions + margin

        return predictions, lower, upper

    def cross_validate(
        self,
        df: pd.DataFrame,
        target_col: str,
        n_splits: int = 5,
        engineer_features: bool = True
    ) -> Dict:
        """
        Perform time-series cross-validation.

        Args:
            df: DataFrame with data
            target_col: Target column
            n_splits: Number of CV folds
            engineer_features: Whether to apply feature engineering

        Returns:
            Cross-validation results
        """
        self.target_col = target_col

        if engineer_features:
            df = self.feature_engineer.engineer_player_features(df, target_col)

        feature_cols = self.feature_engineer.get_feature_columns(df, target_col)
        X, y = self.feature_engineer.prepare_training_data(df, target_col, feature_cols)

        # Align df with X
        df_aligned = df.loc[X.index]

        validator = CrossValidator(n_splits=n_splits)

        # Clone model for CV
        cv_model = self._build_model()

        results = validator.cross_validate_model(
            cv_model, X, y, df_aligned,
            is_classification=self._is_classification()
        )

        validator.print_cv_results(results, self.model_name)

        return results

    def _is_classification(self) -> bool:
        """Override in classification models."""
        return False

    @abstractmethod
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate model-specific metrics."""
        pass

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importances if available.

        Returns:
            DataFrame with feature names and importances
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")

        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_)
        else:
            raise ValueError("Model does not support feature importance")

        importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)

        return importance_df.head(top_n)

    def save(self, filename: Optional[str] = None):
        """Save model to disk."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving")

        if filename is None:
            filename = f"{self.model_name}.pkl"

        filepath = self.model_dir / filename

        save_data = {
            'model': self.model,
            'feature_cols': self.feature_cols,
            'target_col': self.target_col,
            'params': self.params,
            'training_metadata': self.training_metadata,
            'model_name': self.model_name,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"Model saved to {filepath}")

    def load(self, filename: Optional[str] = None):
        """Load model from disk."""
        if filename is None:
            filename = f"{self.model_name}.pkl"

        filepath = self.model_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)

        self.model = save_data['model']
        self.feature_cols = save_data['feature_cols']
        self.target_col = save_data['target_col']
        self.params = save_data['params']
        self.training_metadata = save_data['training_metadata']
        self.is_fitted = True

        print(f"Model loaded from {filepath}")
