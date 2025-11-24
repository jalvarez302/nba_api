"""Statistical models for NBA predictions."""

from nba_api.stats_modeling.models.base import NBABaseModel
from nba_api.stats_modeling.models.points import PointsPredictor, PointsPredictorEnsemble
from nba_api.stats_modeling.models.game_outcome import GameOutcomePredictor
from nba_api.stats_modeling.models.individual_stats import (
    IndividualStatsPredictor,
    ThreePointPredictor,
    FreeThrowPredictor,
    BlocksPredictor,
    FieldGoalPredictor,
)
from nba_api.stats_modeling.models.ensemble import (
    StackedEnsembleRegressor,
    StackedEnsembleClassifier,
    BlendingEnsemble,
    BayesianModelAveraging,
)
from nba_api.stats_modeling.models.props import (
    PlayerPropsPredictor,
    ComboPropsPredictor,
    AlternateLineAnalyzer,
)
from nba_api.stats_modeling.models.spreads import (
    SpreadPredictor,
    TotalsPredictor,
)

__all__ = [
    # Base
    "NBABaseModel",
    # Points
    "PointsPredictor",
    "PointsPredictorEnsemble",
    # Game Outcome
    "GameOutcomePredictor",
    # Individual Stats
    "IndividualStatsPredictor",
    "ThreePointPredictor",
    "FreeThrowPredictor",
    "BlocksPredictor",
    "FieldGoalPredictor",
    # Ensemble
    "StackedEnsembleRegressor",
    "StackedEnsembleClassifier",
    "BlendingEnsemble",
    "BayesianModelAveraging",
    # Props
    "PlayerPropsPredictor",
    "ComboPropsPredictor",
    "AlternateLineAnalyzer",
    # Spreads/Totals
    "SpreadPredictor",
    "TotalsPredictor",
]
