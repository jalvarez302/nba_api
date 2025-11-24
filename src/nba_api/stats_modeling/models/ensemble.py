"""
Advanced Ensemble Models for NBA Prediction

Implements sophisticated ensemble methods used by professional sports betting:
- Stacking ensembles with meta-learners
- Blending with optimized weights
- Bayesian model averaging
- Calibrated probability estimates
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.ensemble import (
    StackingRegressor, StackingClassifier,
    VotingRegressor, VotingClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier,
    RandomForestRegressor, RandomForestClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
)
from sklearn.linear_model import (
    Ridge, RidgeClassifier, LogisticRegression,
    ElasticNet, BayesianRidge, Lasso
)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, log_loss, brier_score_loss
import warnings

# Optional imports
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


class StackedEnsembleRegressor(BaseEstimator, RegressorMixin):
    """
    Professional-grade stacking ensemble for regression.

    Uses multiple diverse base models with a meta-learner to combine predictions.
    This approach captures different aspects of the data and reduces overfitting.
    """

    def __init__(
        self,
        use_neural_net: bool = True,
        use_advanced_boosting: bool = True,
        meta_learner: str = 'ridge',
        cv_folds: int = 5,
        random_state: int = 42
    ):
        self.use_neural_net = use_neural_net
        self.use_advanced_boosting = use_advanced_boosting
        self.meta_learner = meta_learner
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.model = None
        self.base_models_ = None

    def _get_base_models(self) -> List[Tuple[str, Any]]:
        """Create diverse base models for stacking."""
        models = [
            ('hgb', HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.05, max_depth=6,
                random_state=self.random_state
            )),
            ('rf', RandomForestRegressor(
                n_estimators=200, max_depth=10, min_samples_leaf=10,
                random_state=self.random_state, n_jobs=-1
            )),
            ('et', ExtraTreesRegressor(
                n_estimators=200, max_depth=10, min_samples_leaf=10,
                random_state=self.random_state, n_jobs=-1
            )),
            ('ridge', Ridge(alpha=1.0)),
            ('bayesian', BayesianRidge()),
        ]

        if self.use_neural_net:
            models.append(('mlp', MLPRegressor(
                hidden_layer_sizes=(100, 50), max_iter=500,
                early_stopping=True, random_state=self.random_state
            )))

        if self.use_advanced_boosting:
            if HAS_LIGHTGBM:
                models.append(('lgbm', lgb.LGBMRegressor(
                    n_estimators=200, learning_rate=0.05, max_depth=6,
                    random_state=self.random_state, verbose=-1, n_jobs=-1
                )))
            if HAS_XGBOOST:
                models.append(('xgb', xgb.XGBRegressor(
                    n_estimators=200, learning_rate=0.05, max_depth=6,
                    random_state=self.random_state, verbosity=0, n_jobs=-1
                )))

        return models

    def _get_meta_learner(self):
        """Get the meta-learner for combining base model predictions."""
        if self.meta_learner == 'ridge':
            return Ridge(alpha=1.0)
        elif self.meta_learner == 'elastic':
            return ElasticNet(alpha=0.1, l1_ratio=0.5)
        elif self.meta_learner == 'bayesian':
            return BayesianRidge()
        elif self.meta_learner == 'mlp':
            return MLPRegressor(
                hidden_layer_sizes=(50,), max_iter=500,
                early_stopping=True, random_state=self.random_state
            )
        else:
            return Ridge(alpha=1.0)

    def fit(self, X, y):
        """Fit the stacking ensemble."""
        self.base_models_ = self._get_base_models()

        self.model = StackingRegressor(
            estimators=self.base_models_,
            final_estimator=self._get_meta_learner(),
            cv=TimeSeriesSplit(n_splits=self.cv_folds),
            n_jobs=-1,
            passthrough=False  # Only use base model predictions
        )

        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)

    def get_base_model_weights(self) -> Dict[str, float]:
        """Get the learned weights for each base model."""
        if self.model is None:
            return {}

        meta = self.model.final_estimator_
        if hasattr(meta, 'coef_'):
            weights = meta.coef_
            return {name: float(w) for (name, _), w in
                    zip(self.base_models_, weights)}
        return {}


class StackedEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Professional-grade stacking ensemble for classification (win/loss prediction).

    Produces calibrated probability estimates suitable for betting applications.
    """

    def __init__(
        self,
        use_neural_net: bool = True,
        use_advanced_boosting: bool = True,
        calibrate: bool = True,
        cv_folds: int = 5,
        random_state: int = 42
    ):
        self.use_neural_net = use_neural_net
        self.use_advanced_boosting = use_advanced_boosting
        self.calibrate = calibrate
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.model = None
        self.calibrator = None

    def _get_base_models(self) -> List[Tuple[str, Any]]:
        """Create diverse base models."""
        models = [
            ('hgb', HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, max_depth=6,
                random_state=self.random_state
            )),
            ('rf', RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_leaf=10,
                random_state=self.random_state, n_jobs=-1
            )),
            ('et', ExtraTreesClassifier(
                n_estimators=200, max_depth=10, min_samples_leaf=10,
                random_state=self.random_state, n_jobs=-1
            )),
            ('lr', LogisticRegression(C=1.0, max_iter=1000)),
        ]

        if self.use_neural_net:
            models.append(('mlp', MLPClassifier(
                hidden_layer_sizes=(100, 50), max_iter=500,
                early_stopping=True, random_state=self.random_state
            )))

        if self.use_advanced_boosting:
            if HAS_LIGHTGBM:
                models.append(('lgbm', lgb.LGBMClassifier(
                    n_estimators=200, learning_rate=0.05, max_depth=6,
                    random_state=self.random_state, verbose=-1, n_jobs=-1
                )))
            if HAS_XGBOOST:
                models.append(('xgb', xgb.XGBClassifier(
                    n_estimators=200, learning_rate=0.05, max_depth=6,
                    random_state=self.random_state, verbosity=0, n_jobs=-1,
                    use_label_encoder=False, eval_metric='logloss'
                )))

        return models

    def fit(self, X, y):
        """Fit the stacking ensemble with optional calibration."""
        base_models = self._get_base_models()

        self.model = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(C=1.0, max_iter=1000),
            cv=TimeSeriesSplit(n_splits=self.cv_folds),
            stack_method='predict_proba',
            n_jobs=-1
        )

        if self.calibrate:
            # Wrap in calibration for better probability estimates
            self.calibrator = CalibratedClassifierCV(
                self.model, method='isotonic', cv=3
            )
            self.calibrator.fit(X, y)
        else:
            self.model.fit(X, y)

        return self

    def predict(self, X):
        """Predict class labels."""
        if self.calibrate and self.calibrator:
            return self.calibrator.predict(X)
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict calibrated probabilities."""
        if self.calibrate and self.calibrator:
            return self.calibrator.predict_proba(X)
        return self.model.predict_proba(X)


class BlendingEnsemble(BaseEstimator):
    """
    Blending ensemble with optimized weights.

    Uses holdout predictions to learn optimal combination weights,
    avoiding overfitting from nested cross-validation.
    """

    def __init__(
        self,
        models: Optional[List] = None,
        blend_method: str = 'optimize',  # 'optimize', 'average', 'rank'
        holdout_fraction: float = 0.2,
        task: str = 'regression',  # 'regression' or 'classification'
        random_state: int = 42
    ):
        self.models = models
        self.blend_method = blend_method
        self.holdout_fraction = holdout_fraction
        self.task = task
        self.random_state = random_state
        self.weights_ = None
        self.fitted_models_ = None

    def _default_models(self):
        """Get default models based on task."""
        if self.task == 'regression':
            models = [
                HistGradientBoostingRegressor(max_iter=200, random_state=self.random_state),
                RandomForestRegressor(n_estimators=200, random_state=self.random_state),
                Ridge(alpha=1.0),
            ]
            if HAS_LIGHTGBM:
                models.append(lgb.LGBMRegressor(n_estimators=200, verbose=-1))
        else:
            models = [
                HistGradientBoostingClassifier(max_iter=200, random_state=self.random_state),
                RandomForestClassifier(n_estimators=200, random_state=self.random_state),
                LogisticRegression(max_iter=1000),
            ]
            if HAS_LIGHTGBM:
                models.append(lgb.LGBMClassifier(n_estimators=200, verbose=-1))

        return models

    def fit(self, X, y):
        """Fit blending ensemble."""
        if self.models is None:
            self.models = self._default_models()

        n_samples = len(X)
        n_holdout = int(n_samples * self.holdout_fraction)

        # Time-series split: use last portion for blending
        X_train = X.iloc[:-n_holdout] if hasattr(X, 'iloc') else X[:-n_holdout]
        y_train = y.iloc[:-n_holdout] if hasattr(y, 'iloc') else y[:-n_holdout]
        X_blend = X.iloc[-n_holdout:] if hasattr(X, 'iloc') else X[-n_holdout:]
        y_blend = y.iloc[-n_holdout:] if hasattr(y, 'iloc') else y[-n_holdout:]

        # Train base models
        self.fitted_models_ = []
        blend_predictions = []

        for model in self.models:
            fitted = clone(model)
            fitted.fit(X_train, y_train)
            self.fitted_models_.append(fitted)

            if self.task == 'classification' and hasattr(fitted, 'predict_proba'):
                pred = fitted.predict_proba(X_blend)[:, 1]
            else:
                pred = fitted.predict(X_blend)
            blend_predictions.append(pred)

        blend_predictions = np.column_stack(blend_predictions)

        # Learn weights
        if self.blend_method == 'optimize':
            self.weights_ = self._optimize_weights(blend_predictions, y_blend)
        elif self.blend_method == 'rank':
            self.weights_ = self._rank_weights(blend_predictions, y_blend)
        else:
            self.weights_ = np.ones(len(self.models)) / len(self.models)

        # Retrain on full data
        self.fitted_models_ = []
        for model in self.models:
            fitted = clone(model)
            fitted.fit(X, y)
            self.fitted_models_.append(fitted)

        return self

    def _optimize_weights(self, predictions, y_true) -> np.ndarray:
        """Optimize blending weights using constrained optimization."""
        from scipy.optimize import minimize

        n_models = predictions.shape[1]

        def objective(weights):
            blended = np.dot(predictions, weights)
            if self.task == 'regression':
                return mean_squared_error(y_true, blended)
            else:
                return log_loss(y_true, np.clip(blended, 1e-7, 1-1e-7))

        # Constraints: weights sum to 1, all non-negative
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_models)]

        result = minimize(
            objective,
            x0=np.ones(n_models) / n_models,
            bounds=bounds,
            constraints=constraints,
            method='SLSQP'
        )

        return result.x

    def _rank_weights(self, predictions, y_true) -> np.ndarray:
        """Assign weights based on individual model performance."""
        scores = []
        for i in range(predictions.shape[1]):
            if self.task == 'regression':
                score = -mean_squared_error(y_true, predictions[:, i])
            else:
                score = -log_loss(y_true, np.clip(predictions[:, i], 1e-7, 1-1e-7))
            scores.append(score)

        # Convert scores to weights (softmax-like)
        scores = np.array(scores)
        weights = np.exp(scores - np.max(scores))
        weights /= weights.sum()

        return weights

    def predict(self, X):
        """Make blended predictions."""
        predictions = []
        for model in self.fitted_models_:
            if self.task == 'classification' and hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.predict(X)
            predictions.append(pred)

        predictions = np.column_stack(predictions)
        blended = np.dot(predictions, self.weights_)

        if self.task == 'classification':
            return (blended > 0.5).astype(int)
        return blended

    def predict_proba(self, X):
        """Get blended probabilities (classification only)."""
        if self.task != 'classification':
            raise ValueError("predict_proba only for classification")

        predictions = []
        for model in self.fitted_models_:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.predict(X)
            predictions.append(pred)

        predictions = np.column_stack(predictions)
        blended = np.dot(predictions, self.weights_)

        return np.column_stack([1 - blended, blended])


class BayesianModelAveraging(BaseEstimator):
    """
    Bayesian Model Averaging for uncertainty quantification.

    Provides prediction intervals that account for both:
    - Model uncertainty (which model is best)
    - Parameter uncertainty (within each model)
    """

    def __init__(
        self,
        models: Optional[List] = None,
        n_bootstrap: int = 100,
        task: str = 'regression',
        random_state: int = 42
    ):
        self.models = models
        self.n_bootstrap = n_bootstrap
        self.task = task
        self.random_state = random_state
        self.model_weights_ = None
        self.fitted_models_ = None
        self.bootstrap_models_ = None

    def fit(self, X, y):
        """Fit with Bayesian model averaging."""
        if self.models is None:
            if self.task == 'regression':
                self.models = [
                    BayesianRidge(),
                    HistGradientBoostingRegressor(max_iter=100),
                    Ridge(alpha=1.0),
                ]
            else:
                self.models = [
                    LogisticRegression(max_iter=1000),
                    HistGradientBoostingClassifier(max_iter=100),
                ]

        X_arr = X.values if hasattr(X, 'values') else X
        y_arr = y.values if hasattr(y, 'values') else y

        # Compute model weights using BIC approximation
        self.fitted_models_ = []
        log_likelihoods = []

        tscv = TimeSeriesSplit(n_splits=3)

        for model in self.models:
            fitted = clone(model)

            # Cross-validate to estimate likelihood
            cv_preds = []
            cv_true = []

            for train_idx, val_idx in tscv.split(X_arr):
                m = clone(model)
                m.fit(X_arr[train_idx], y_arr[train_idx])

                if self.task == 'classification' and hasattr(m, 'predict_proba'):
                    pred = m.predict_proba(X_arr[val_idx])[:, 1]
                else:
                    pred = m.predict(X_arr[val_idx])

                cv_preds.extend(pred)
                cv_true.extend(y_arr[val_idx])

            cv_preds = np.array(cv_preds)
            cv_true = np.array(cv_true)

            if self.task == 'regression':
                mse = mean_squared_error(cv_true, cv_preds)
                ll = -len(cv_true) * np.log(mse + 1e-8) / 2
            else:
                ll = -log_loss(cv_true, np.clip(cv_preds, 1e-7, 1-1e-7))

            log_likelihoods.append(ll)

            # Fit on full data
            fitted.fit(X_arr, y_arr)
            self.fitted_models_.append(fitted)

        # Convert to weights using softmax
        log_likelihoods = np.array(log_likelihoods)
        self.model_weights_ = np.exp(log_likelihoods - np.max(log_likelihoods))
        self.model_weights_ /= self.model_weights_.sum()

        # Bootstrap for uncertainty estimation
        self._fit_bootstrap(X_arr, y_arr)

        return self

    def _fit_bootstrap(self, X, y):
        """Fit bootstrap models for uncertainty estimation."""
        np.random.seed(self.random_state)
        self.bootstrap_models_ = []

        n_samples = len(X)

        for _ in range(self.n_bootstrap):
            # Bootstrap sample (respecting time order somewhat)
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            # Fit weighted random model
            model_idx = np.random.choice(
                len(self.models),
                p=self.model_weights_
            )
            model = clone(self.models[model_idx])
            model.fit(X_boot, y_boot)
            self.bootstrap_models_.append(model)

    def predict(self, X):
        """Make averaged predictions."""
        predictions = []
        for model, weight in zip(self.fitted_models_, self.model_weights_):
            if self.task == 'classification' and hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.predict(X)
            predictions.append(pred * weight)

        return np.sum(predictions, axis=0)

    def predict_with_uncertainty(
        self,
        X,
        confidence: float = 0.9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with uncertainty intervals.

        Returns:
            Tuple of (mean_prediction, lower_bound, upper_bound)
        """
        X_arr = X.values if hasattr(X, 'values') else X

        # Get predictions from all bootstrap models
        all_preds = []
        for model in self.bootstrap_models_:
            if self.task == 'classification' and hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X_arr)[:, 1]
            else:
                pred = model.predict(X_arr)
            all_preds.append(pred)

        all_preds = np.array(all_preds)

        mean_pred = np.mean(all_preds, axis=0)
        lower = np.percentile(all_preds, (1 - confidence) / 2 * 100, axis=0)
        upper = np.percentile(all_preds, (1 + confidence) / 2 * 100, axis=0)

        return mean_pred, lower, upper
