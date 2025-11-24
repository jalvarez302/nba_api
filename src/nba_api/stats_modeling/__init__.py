"""
NBA Statistics Modeling Package - Professional Sports Betting Analytics

A comprehensive machine learning platform for NBA predictions with features
used by professional sports betting operations:

DATA:
- NBADataStorage: SQLite + Parquet hybrid storage
- NBADataCollector: Rate-limited API data collection

MODELS:
- PointsPredictor: Player points prediction
- GameOutcomePredictor: Win/loss probability
- SpreadPredictor: Point spread prediction
- TotalsPredictor: Over/under totals
- PlayerPropsPredictor: Player prop betting
- IndividualStatsPredictor: Multi-stat prediction
- Ensemble models: Stacking, blending, Bayesian averaging

BETTING:
- OddsConverter: American/Decimal/Fractional conversion
- BankrollManager: Kelly criterion and bet sizing
- Backtester: Strategy backtesting framework

Example usage:
    from nba_api.stats_modeling import NBAPredictor

    predictor = NBAPredictor()
    predictor.collect_data(seasons=['2023-24', '2024-25'])
    predictor.train_all_models()

    points = predictor.predict_points(player_id=2544)
    game = predictor.predict_game(home_team_id=1610612747, away_team_id=1610612744)
"""

__version__ = "2.0.0"

# Data
from nba_api.stats_modeling.data.storage import NBADataStorage
from nba_api.stats_modeling.data.collector import NBADataCollector

# Main predictor
from nba_api.stats_modeling.predictor import NBAPredictor, create_nba_predictor

# Core models
from nba_api.stats_modeling.models.points import PointsPredictor, PointsPredictorEnsemble
from nba_api.stats_modeling.models.game_outcome import GameOutcomePredictor
from nba_api.stats_modeling.models.individual_stats import (
    IndividualStatsPredictor,
    ThreePointPredictor,
    FreeThrowPredictor,
    BlocksPredictor,
    FieldGoalPredictor,
)

# Advanced models
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

# Betting utilities
from nba_api.stats_modeling.betting.odds import (
    OddsConverter,
    american_to_decimal,
    decimal_to_american,
    implied_probability,
    calculate_ev,
    calculate_vig,
    no_vig_odds,
)
from nba_api.stats_modeling.betting.bankroll import (
    BankrollManager,
    kelly_criterion,
    fractional_kelly,
)
from nba_api.stats_modeling.betting.backtesting import (
    Backtester,
    BacktestResult,
)

# Feature engineering
from nba_api.stats_modeling.utils.features import FeatureEngineer
from nba_api.stats_modeling.utils.validation import CrossValidator

__all__ = [
    # Version
    "__version__",
    # Data
    "NBADataStorage",
    "NBADataCollector",
    # Main interface
    "NBAPredictor",
    "create_nba_predictor",
    # Core models
    "PointsPredictor",
    "PointsPredictorEnsemble",
    "GameOutcomePredictor",
    "IndividualStatsPredictor",
    "ThreePointPredictor",
    "FreeThrowPredictor",
    "BlocksPredictor",
    "FieldGoalPredictor",
    # Advanced models
    "StackedEnsembleRegressor",
    "StackedEnsembleClassifier",
    "BlendingEnsemble",
    "BayesianModelAveraging",
    "PlayerPropsPredictor",
    "ComboPropsPredictor",
    "AlternateLineAnalyzer",
    "SpreadPredictor",
    "TotalsPredictor",
    # Betting utilities
    "OddsConverter",
    "american_to_decimal",
    "decimal_to_american",
    "implied_probability",
    "calculate_ev",
    "calculate_vig",
    "no_vig_odds",
    "BankrollManager",
    "kelly_criterion",
    "fractional_kelly",
    "Backtester",
    "BacktestResult",
    # Utils
    "FeatureEngineer",
    "CrossValidator",
]
