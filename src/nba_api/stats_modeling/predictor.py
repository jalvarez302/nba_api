"""
Unified NBA Statistics Predictor

High-level interface that combines all prediction models and data collection
for easy end-to-end NBA statistics prediction.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import warnings

import pandas as pd
import numpy as np

from nba_api.stats_modeling.data.storage import NBADataStorage
from nba_api.stats_modeling.data.collector import NBADataCollector
from nba_api.stats_modeling.models.points import PointsPredictor, PointsPredictorEnsemble
from nba_api.stats_modeling.models.game_outcome import GameOutcomePredictor
from nba_api.stats_modeling.models.individual_stats import (
    IndividualStatsPredictor,
    ThreePointPredictor,
    FreeThrowPredictor,
    BlocksPredictor,
    FieldGoalPredictor,
)
from nba_api.stats_modeling.utils.features import FeatureEngineer


class NBAPredictor:
    """
    Unified interface for NBA statistics prediction.

    Provides:
    - Automatic data collection and storage
    - Model training with best practices
    - Easy prediction API
    - Model persistence

    Example usage:
    ```python
    predictor = NBAPredictor()

    # Collect data and train models
    predictor.collect_data(seasons=['2023-24', '2024-25'])
    predictor.train_all_models()

    # Make predictions
    points = predictor.predict_points(player_id=2544)  # LeBron James
    outcome = predictor.predict_game(home_team_id=1610612747, away_team_id=1610612744)
    all_stats = predictor.predict_all_stats(player_id=2544)
    ```
    """

    def __init__(
        self,
        data_dir: str = "./nba_data",
        model_dir: str = "./nba_models",
        backend: str = 'sklearn'
    ):
        """
        Initialize the NBA Predictor.

        Args:
            data_dir: Directory for data storage
            model_dir: Directory for model persistence
            backend: ML backend ('sklearn', 'lightgbm', 'xgboost')
        """
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend

        # Initialize components
        self.storage = NBADataStorage(data_dir)
        self.collector = NBADataCollector(storage=self.storage)
        self.feature_engineer = FeatureEngineer()

        # Models (initialized on first use)
        self._points_model = None
        self._game_outcome_model = None
        self._stats_model = None
        self._specialized_models = {}

        # Cached data
        self._player_logs = None
        self._team_logs = None

    # ===================
    # Data Collection
    # ===================

    def collect_data(
        self,
        seasons: Optional[List[str]] = None,
        include_box_scores: bool = False
    ) -> Dict:
        """
        Collect NBA data for modeling.

        Args:
            seasons: List of seasons to collect (e.g., ['2023-24', '2024-25'])
            include_box_scores: Whether to collect detailed box scores (slow)

        Returns:
            Collection statistics
        """
        if seasons is None:
            seasons = ['2024-25', '2023-24', '2022-23']

        print("Starting data collection...")
        stats = self.collector.collect_all_data(
            seasons=seasons,
            include_box_scores=include_box_scores
        )

        # Clear cached data
        self._player_logs = None
        self._team_logs = None

        return stats

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load collected data into memory.

        Returns:
            Dictionary with DataFrames for player_logs, team_logs
        """
        try:
            self._player_logs = self.storage.load_dataset("league_player_game_logs")
            print(f"Loaded {len(self._player_logs)} player game log entries")
        except (ValueError, FileNotFoundError):
            print("Player game logs not found. Run collect_data() first.")
            self._player_logs = pd.DataFrame()

        try:
            self._team_logs = self.storage.load_dataset("team_game_logs")
            print(f"Loaded {len(self._team_logs)} team game log entries")
        except (ValueError, FileNotFoundError):
            print("Team game logs not found. Run collect_data() first.")
            self._team_logs = pd.DataFrame()

        return {
            'player_logs': self._player_logs,
            'team_logs': self._team_logs,
        }

    # ===================
    # Model Training
    # ===================

    def train_points_model(
        self,
        use_ensemble: bool = False,
        **kwargs
    ) -> Dict:
        """
        Train the points prediction model.

        Args:
            use_ensemble: Whether to use ensemble of models
            **kwargs: Additional training parameters

        Returns:
            Training metrics
        """
        if self._player_logs is None or len(self._player_logs) == 0:
            self.load_data()

        if len(self._player_logs) == 0:
            raise ValueError("No player data available. Run collect_data() first.")

        print("\nTraining points prediction model...")

        if use_ensemble:
            self._points_model = PointsPredictorEnsemble()
            self._points_model.fit(self._player_logs, target_col='PTS')
            return {'status': 'ensemble trained'}
        else:
            self._points_model = PointsPredictor(
                backend=self.backend,
                model_dir=str(self.model_dir)
            )
            return self._points_model.fit(
                self._player_logs,
                target_col='PTS',
                **kwargs
            )

    def train_game_outcome_model(self, **kwargs) -> Dict:
        """
        Train the game outcome prediction model.

        Returns:
            Training metrics
        """
        if self._team_logs is None or len(self._team_logs) == 0:
            self.load_data()

        if len(self._team_logs) == 0:
            raise ValueError("No team data available. Run collect_data() first.")

        print("\nTraining game outcome prediction model...")

        self._game_outcome_model = GameOutcomePredictor(
            backend=self.backend,
            model_dir=str(self.model_dir)
        )

        return self._game_outcome_model.fit(
            self._team_logs,
            target_col='WIN',
            **kwargs
        )

    def train_individual_stats_model(
        self,
        targets: Optional[List[str]] = None,
        **kwargs
    ) -> Dict:
        """
        Train the individual statistics prediction model.

        Args:
            targets: Stats to predict (default: FG3M, FTM, BLK, FGM)

        Returns:
            Training metrics
        """
        if self._player_logs is None or len(self._player_logs) == 0:
            self.load_data()

        if len(self._player_logs) == 0:
            raise ValueError("No player data available. Run collect_data() first.")

        print("\nTraining individual stats prediction model...")

        self._stats_model = IndividualStatsPredictor(
            targets=targets,
            backend=self.backend,
            model_dir=str(self.model_dir)
        )

        return self._stats_model.fit(
            self._player_logs,
            **kwargs
        )

    def train_all_models(self, **kwargs) -> Dict[str, Dict]:
        """
        Train all prediction models.

        Returns:
            Dictionary of training results for each model
        """
        results = {}

        print("=" * 60)
        print("Training All NBA Prediction Models")
        print("=" * 60)

        try:
            results['points'] = self.train_points_model(**kwargs)
            print(f"Points model trained: RMSE = {results['points']['validation_metrics'].get('rmse', 'N/A'):.2f}")
        except Exception as e:
            print(f"Failed to train points model: {e}")
            results['points'] = {'error': str(e)}

        try:
            results['game_outcome'] = self.train_game_outcome_model(**kwargs)
            print(f"Game outcome model trained: Accuracy = {results['game_outcome']['validation_metrics'].get('accuracy', 'N/A'):.2%}")
        except Exception as e:
            print(f"Failed to train game outcome model: {e}")
            results['game_outcome'] = {'error': str(e)}

        try:
            results['individual_stats'] = self.train_individual_stats_model(**kwargs)
            print("Individual stats model trained")
        except Exception as e:
            print(f"Failed to train individual stats model: {e}")
            results['individual_stats'] = {'error': str(e)}

        print("\n" + "=" * 60)
        print("All models trained successfully!")
        print("=" * 60)

        return results

    # ===================
    # Predictions
    # ===================

    def predict_points(
        self,
        player_id: int,
        n_games_context: int = 20
    ) -> Dict:
        """
        Predict points for a specific player.

        Args:
            player_id: NBA player ID
            n_games_context: Number of recent games for context

        Returns:
            Dictionary with prediction and confidence interval
        """
        if self._points_model is None:
            raise RuntimeError("Points model not trained. Call train_points_model() first.")

        if self._player_logs is None:
            self.load_data()

        return self._points_model.predict_player(
            player_id=player_id,
            game_logs_df=self._player_logs,
            n_games_context=n_games_context
        )

    def predict_game(
        self,
        home_team_id: int,
        away_team_id: int,
        n_games_context: int = 10
    ) -> Dict:
        """
        Predict game outcome.

        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            n_games_context: Number of recent games for context

        Returns:
            Dictionary with win probabilities
        """
        if self._game_outcome_model is None:
            raise RuntimeError("Game outcome model not trained. Call train_game_outcome_model() first.")

        if self._team_logs is None:
            self.load_data()

        home_logs = self._team_logs[self._team_logs['TEAM_ID'] == home_team_id]
        away_logs = self._team_logs[self._team_logs['TEAM_ID'] == away_team_id]

        return self._game_outcome_model.predict_game(
            home_team_logs=home_logs,
            away_team_logs=away_logs,
            n_games_context=n_games_context
        )

    def predict_all_stats(
        self,
        player_id: int,
        n_games_context: int = 20
    ) -> Dict:
        """
        Predict all statistics for a player.

        Args:
            player_id: NBA player ID
            n_games_context: Number of recent games for context

        Returns:
            Dictionary with all stat predictions
        """
        if self._stats_model is None:
            raise RuntimeError("Stats model not trained. Call train_individual_stats_model() first.")

        if self._player_logs is None:
            self.load_data()

        return self._stats_model.predict_player_stats(
            player_id=player_id,
            game_logs_df=self._player_logs,
            n_games_context=n_games_context
        )

    def predict_three_pointers(self, player_id: int, **kwargs) -> Dict:
        """Predict three-pointers made for a player."""
        if 'three_point' not in self._specialized_models:
            self._specialized_models['three_point'] = ThreePointPredictor(backend=self.backend)
            if self._player_logs is not None and len(self._player_logs) > 0:
                self._specialized_models['three_point'].fit(self._player_logs)

        return self._specialized_models['three_point'].predict_player_stats(
            player_id=player_id,
            game_logs_df=self._player_logs,
            **kwargs
        )

    def predict_free_throws(self, player_id: int, **kwargs) -> Dict:
        """Predict free throws made for a player."""
        if 'free_throw' not in self._specialized_models:
            self._specialized_models['free_throw'] = FreeThrowPredictor(backend=self.backend)
            if self._player_logs is not None and len(self._player_logs) > 0:
                self._specialized_models['free_throw'].fit(self._player_logs)

        return self._specialized_models['free_throw'].predict_player_stats(
            player_id=player_id,
            game_logs_df=self._player_logs,
            **kwargs
        )

    def predict_blocks(self, player_id: int, **kwargs) -> Dict:
        """Predict blocks for a player."""
        if 'blocks' not in self._specialized_models:
            self._specialized_models['blocks'] = BlocksPredictor(backend=self.backend)
            if self._player_logs is not None and len(self._player_logs) > 0:
                self._specialized_models['blocks'].fit(self._player_logs)

        return self._specialized_models['blocks'].predict_player_stats(
            player_id=player_id,
            game_logs_df=self._player_logs,
            **kwargs
        )

    def predict_field_goals(self, player_id: int, **kwargs) -> Dict:
        """Predict field goals made for a player."""
        if 'field_goal' not in self._specialized_models:
            self._specialized_models['field_goal'] = FieldGoalPredictor(backend=self.backend)
            if self._player_logs is not None and len(self._player_logs) > 0:
                self._specialized_models['field_goal'].fit(self._player_logs)

        return self._specialized_models['field_goal'].predict_player_stats(
            player_id=player_id,
            game_logs_df=self._player_logs,
            **kwargs
        )

    # ===================
    # Model Persistence
    # ===================

    def save_all_models(self):
        """Save all trained models to disk."""
        if self._points_model is not None and hasattr(self._points_model, 'save'):
            self._points_model.save()
            print("Points model saved")

        if self._game_outcome_model is not None:
            self._game_outcome_model.save()
            print("Game outcome model saved")

        if self._stats_model is not None:
            self._stats_model.save()
            print("Individual stats model saved")

    def load_all_models(self):
        """Load all models from disk."""
        try:
            self._points_model = PointsPredictor(
                backend=self.backend,
                model_dir=str(self.model_dir)
            )
            self._points_model.load()
            print("Points model loaded")
        except FileNotFoundError:
            print("Points model not found")

        try:
            self._game_outcome_model = GameOutcomePredictor(
                backend=self.backend,
                model_dir=str(self.model_dir)
            )
            self._game_outcome_model.load()
            print("Game outcome model loaded")
        except FileNotFoundError:
            print("Game outcome model not found")

        try:
            self._stats_model = IndividualStatsPredictor(
                backend=self.backend,
                model_dir=str(self.model_dir)
            )
            self._stats_model.load()
            print("Individual stats model loaded")
        except FileNotFoundError:
            print("Individual stats model not found")

    # ===================
    # Utilities
    # ===================

    def get_player_id(self, player_name: str) -> Optional[int]:
        """
        Look up player ID by name.

        Args:
            player_name: Player name (partial match supported)

        Returns:
            Player ID or None if not found
        """
        from nba_api.stats.static import players

        matches = players.find_players_by_full_name(player_name, safe_search=True)

        if not matches:
            print(f"No players found matching '{player_name}'")
            return None

        if len(matches) > 1:
            print(f"Multiple players found matching '{player_name}':")
            for p in matches[:5]:
                print(f"  - {p['full_name']} (ID: {p['id']})")
            print("Using first match.")

        return matches[0]['id']

    def get_team_id(self, team_name: str) -> Optional[int]:
        """
        Look up team ID by name or abbreviation.

        Args:
            team_name: Team name or abbreviation

        Returns:
            Team ID or None if not found
        """
        from nba_api.stats.static import teams

        all_teams = teams.get_teams()

        for team in all_teams:
            if (team_name.lower() in team['full_name'].lower() or
                team_name.upper() == team['abbreviation']):
                return team['id']

        print(f"Team not found: {team_name}")
        return None

    def get_storage_stats(self) -> Dict:
        """Get statistics about stored data."""
        return self.storage.get_storage_stats()

    def list_datasets(self) -> pd.DataFrame:
        """List all stored datasets."""
        return self.storage.list_datasets()


# Convenience function for quick setup
def create_nba_predictor(
    collect_seasons: Optional[List[str]] = None,
    train_models: bool = True,
    **kwargs
) -> NBAPredictor:
    """
    Create and optionally set up an NBAPredictor.

    Args:
        collect_seasons: Seasons to collect data for (None to skip collection)
        train_models: Whether to train models after collection

    Returns:
        Configured NBAPredictor instance
    """
    predictor = NBAPredictor(**kwargs)

    if collect_seasons:
        predictor.collect_data(seasons=collect_seasons)

        if train_models:
            predictor.train_all_models()

    return predictor
