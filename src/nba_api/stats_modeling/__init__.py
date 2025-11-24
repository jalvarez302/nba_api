"""
NBA Statistics Modeling Package

This package provides tools for:
- Collecting and storing NBA data for statistical modeling
- Feature engineering for ML models
- Statistical models for predicting player and game statistics

Example usage:
    from nba_api.stats_modeling import NBAPredictor

    # Create predictor and collect data
    predictor = NBAPredictor()
    predictor.collect_data(seasons=['2023-24', '2024-25'])

    # Train models
    predictor.train_all_models()

    # Make predictions
    points = predictor.predict_points(player_id=2544)  # LeBron James
    game = predictor.predict_game(home_team_id=1610612747, away_team_id=1610612744)
"""

__version__ = "1.0.0"

from nba_api.stats_modeling.data.storage import NBADataStorage
from nba_api.stats_modeling.data.collector import NBADataCollector
from nba_api.stats_modeling.predictor import NBAPredictor, create_nba_predictor
from nba_api.stats_modeling.models.points import PointsPredictor
from nba_api.stats_modeling.models.game_outcome import GameOutcomePredictor
from nba_api.stats_modeling.models.individual_stats import (
    IndividualStatsPredictor,
    ThreePointPredictor,
    FreeThrowPredictor,
    BlocksPredictor,
    FieldGoalPredictor,
)

__all__ = [
    "NBADataStorage",
    "NBADataCollector",
    "NBAPredictor",
    "create_nba_predictor",
    "PointsPredictor",
    "GameOutcomePredictor",
    "IndividualStatsPredictor",
    "ThreePointPredictor",
    "FreeThrowPredictor",
    "BlocksPredictor",
    "FieldGoalPredictor",
]
