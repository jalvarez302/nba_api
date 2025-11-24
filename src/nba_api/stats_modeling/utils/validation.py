"""
Cross-validation utilities for NBA statistics modeling.

Provides time-series aware cross-validation to prevent data leakage.
"""

from typing import Dict, List, Optional, Tuple, Generator
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss
)


class CrossValidator:
    """
    Time-series aware cross-validation for sports predictions.

    Uses expanding window validation to simulate real-world prediction:
    - Training on past data
    - Predicting future games
    - No future data leakage
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        gap: int = 0
    ):
        """
        Initialize the cross-validator.

        Args:
            n_splits: Number of cross-validation folds
            test_size: Size of test set in each fold (None for auto)
            gap: Number of samples to skip between train and test
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap

    def time_series_split(
        self,
        df: pd.DataFrame,
        date_col: str = 'GAME_DATE'
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """
        Generate time-series train/test splits.

        Args:
            df: DataFrame to split (must be sorted by date)
            date_col: Name of date column

        Yields:
            Tuple of (train_df, test_df)
        """
        df = df.sort_values(date_col).reset_index(drop=True)

        tscv = TimeSeriesSplit(
            n_splits=self.n_splits,
            test_size=self.test_size,
            gap=self.gap
        )

        for train_idx, test_idx in tscv.split(df):
            yield df.iloc[train_idx], df.iloc[test_idx]

    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate regression predictions.

        Returns:
            Dictionary of metrics
        """
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100,
        }

    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate classification predictions.

        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
        }

        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
            except ValueError:
                pass  # Not enough classes

        return metrics

    def cross_validate_model(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        df: pd.DataFrame,
        date_col: str = 'GAME_DATE',
        is_classification: bool = False
    ) -> Dict[str, List[float]]:
        """
        Perform full cross-validation on a model.

        Args:
            model: Model with fit/predict interface
            X: Feature DataFrame
            y: Target Series
            df: Original DataFrame with date column
            date_col: Date column name
            is_classification: Whether this is a classification task

        Returns:
            Dictionary of metric lists across folds
        """
        # Ensure X and df are aligned
        df_aligned = df.loc[X.index].copy()
        df_aligned['_X_idx'] = range(len(df_aligned))

        all_metrics = []

        for fold_idx, (train_df, test_df) in enumerate(self.time_series_split(df_aligned, date_col)):
            train_idx = train_df['_X_idx'].values
            test_idx = test_df['_X_idx'].values

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            # Fit and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Evaluate
            if is_classification:
                y_pred_proba = None
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                metrics = self.evaluate_classification(y_test, y_pred, y_pred_proba)
            else:
                metrics = self.evaluate_regression(y_test, y_pred)

            metrics['fold'] = fold_idx
            all_metrics.append(metrics)

        # Aggregate results
        result = {}
        for key in all_metrics[0].keys():
            if key != 'fold':
                values = [m[key] for m in all_metrics]
                result[f'{key}_mean'] = np.mean(values)
                result[f'{key}_std'] = np.std(values)
                result[f'{key}_values'] = values

        return result

    def print_cv_results(self, results: Dict, model_name: str = "Model"):
        """Pretty print cross-validation results."""
        print(f"\n{'='*50}")
        print(f"Cross-Validation Results: {model_name}")
        print('='*50)

        for key, value in results.items():
            if key.endswith('_mean'):
                metric_name = key.replace('_mean', '')
                std = results.get(f'{metric_name}_std', 0)
                print(f"  {metric_name}: {value:.4f} (+/- {std:.4f})")
