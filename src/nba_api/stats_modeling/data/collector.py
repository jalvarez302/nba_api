"""
NBA Data Collector Module

Collects data from the NBA API and stores it efficiently for statistical modeling.
Handles rate limiting, pagination, and data validation.
"""

import time
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import warnings

import pandas as pd
import numpy as np

from nba_api.stats.endpoints import (
    PlayerGameLog,
    LeagueGameLog,
    CommonPlayerInfo,
    CommonTeamRoster,
    TeamGameLog,
    BoxScoreTraditionalV2,
    BoxScoreAdvancedV2,
    LeagueStandings,
    PlayerCareerStats,
    TeamYearByYearStats,
    ScoreboardV2,
)
from nba_api.stats.static import players, teams
from nba_api.stats_modeling.data.storage import NBADataStorage


class NBADataCollector:
    """
    Collects NBA data from the API and stores it for statistical modeling.

    Features:
    - Rate limiting to avoid API throttling
    - Automatic retry with exponential backoff
    - Progress tracking for long-running collections
    - Data validation and cleaning
    """

    # API rate limiting settings
    DEFAULT_DELAY = 0.6  # seconds between requests
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0  # base delay for exponential backoff

    def __init__(
        self,
        storage: Optional[NBADataStorage] = None,
        data_dir: str = "./nba_data",
        request_delay: float = DEFAULT_DELAY
    ):
        """
        Initialize the data collector.

        Args:
            storage: NBADataStorage instance (creates one if not provided)
            data_dir: Directory for data storage
            request_delay: Delay between API requests (seconds)
        """
        self.storage = storage or NBADataStorage(data_dir)
        self.request_delay = request_delay
        self._last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request_time = time.time()

    def _make_request(self, endpoint_class, **kwargs):
        """
        Make an API request with retry logic.

        Args:
            endpoint_class: The endpoint class to instantiate
            **kwargs: Arguments to pass to the endpoint

        Returns:
            The endpoint instance
        """
        for attempt in range(self.MAX_RETRIES):
            try:
                self._rate_limit()
                return endpoint_class(**kwargs)
            except Exception as e:
                if attempt == self.MAX_RETRIES - 1:
                    raise
                delay = self.RETRY_DELAY * (2 ** attempt)
                warnings.warn(f"Request failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                time.sleep(delay)

    # ===================
    # Player Data Collection
    # ===================

    def collect_all_players(self, active_only: bool = False) -> pd.DataFrame:
        """
        Collect basic player information from static data.

        Args:
            active_only: If True, only collect active players

        Returns:
            DataFrame with player information
        """
        if active_only:
            player_list = players.get_active_players()
        else:
            player_list = players.get_players()

        df = pd.DataFrame(player_list)

        # Map column names
        df = df.rename(columns={
            'id': 'player_id',
            'full_name': 'full_name',
            'first_name': 'first_name',
            'last_name': 'last_name',
            'is_active': 'is_active',
        })

        # Save to storage
        self.storage.save_players(df)

        return df

    def collect_player_game_logs(
        self,
        player_ids: Optional[List[int]] = None,
        seasons: Optional[List[str]] = None,
        season_type: str = "Regular Season"
    ) -> pd.DataFrame:
        """
        Collect player game logs for statistical modeling.

        Args:
            player_ids: List of player IDs (None for all active players)
            seasons: List of seasons (e.g., ["2023-24", "2022-23"])
            season_type: "Regular Season" or "Playoffs"

        Returns:
            DataFrame with player game logs
        """
        if player_ids is None:
            active_players = players.get_active_players()
            player_ids = [p['id'] for p in active_players]

        if seasons is None:
            seasons = ["2024-25", "2023-24", "2022-23"]

        all_logs = []
        total = len(player_ids) * len(seasons)
        processed = 0

        for player_id in player_ids:
            for season in seasons:
                try:
                    endpoint = self._make_request(
                        PlayerGameLog,
                        player_id=player_id,
                        season=season,
                        season_type_all_star=season_type
                    )
                    df = endpoint.get_data_frames()[0]
                    if len(df) > 0:
                        df['SEASON'] = season
                        all_logs.append(df)

                except Exception as e:
                    warnings.warn(f"Failed to get logs for player {player_id}, season {season}: {e}")

                processed += 1
                if processed % 50 == 0:
                    print(f"Progress: {processed}/{total} ({100*processed/total:.1f}%)")

        if not all_logs:
            return pd.DataFrame()

        result_df = pd.concat(all_logs, ignore_index=True)

        # Save as dataset
        self.storage.save_dataset(
            result_df,
            dataset_name="player_game_logs",
            description=f"Player game logs for seasons {seasons}",
            season=",".join(seasons),
            data_type="player_stats"
        )

        return result_df

    def collect_player_career_stats(
        self,
        player_ids: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Collect career statistics for players."""
        if player_ids is None:
            active_players = players.get_active_players()
            player_ids = [p['id'] for p in active_players]

        all_stats = []

        for i, player_id in enumerate(player_ids):
            try:
                endpoint = self._make_request(
                    PlayerCareerStats,
                    player_id=player_id,
                    per_mode36="PerGame"
                )
                df = endpoint.get_data_frames()[0]  # Regular season totals
                if len(df) > 0:
                    all_stats.append(df)

            except Exception as e:
                warnings.warn(f"Failed to get career stats for player {player_id}: {e}")

            if (i + 1) % 50 == 0:
                print(f"Progress: {i + 1}/{len(player_ids)}")

        if not all_stats:
            return pd.DataFrame()

        result_df = pd.concat(all_stats, ignore_index=True)

        self.storage.save_dataset(
            result_df,
            dataset_name="player_career_stats",
            description="Career statistics for players",
            data_type="player_stats"
        )

        return result_df

    # ===================
    # Team Data Collection
    # ===================

    def collect_all_teams(self) -> pd.DataFrame:
        """Collect all NBA teams."""
        team_list = teams.get_teams()
        df = pd.DataFrame(team_list)

        df = df.rename(columns={
            'id': 'team_id',
            'full_name': 'full_name',
            'abbreviation': 'abbreviation',
            'city': 'city',
            'state': 'state',
        })

        self.storage.save_teams(df)
        return df

    def collect_team_game_logs(
        self,
        team_ids: Optional[List[int]] = None,
        seasons: Optional[List[str]] = None,
        season_type: str = "Regular Season"
    ) -> pd.DataFrame:
        """Collect team game logs."""
        if team_ids is None:
            all_teams = teams.get_teams()
            team_ids = [t['id'] for t in all_teams]

        if seasons is None:
            seasons = ["2024-25", "2023-24", "2022-23"]

        all_logs = []

        for team_id in team_ids:
            for season in seasons:
                try:
                    endpoint = self._make_request(
                        TeamGameLog,
                        team_id=team_id,
                        season=season,
                        season_type_all_star=season_type
                    )
                    df = endpoint.get_data_frames()[0]
                    if len(df) > 0:
                        df['SEASON'] = season
                        all_logs.append(df)

                except Exception as e:
                    warnings.warn(f"Failed to get logs for team {team_id}, season {season}: {e}")

        if not all_logs:
            return pd.DataFrame()

        result_df = pd.concat(all_logs, ignore_index=True)

        self.storage.save_dataset(
            result_df,
            dataset_name="team_game_logs",
            description=f"Team game logs for seasons {seasons}",
            season=",".join(seasons),
            data_type="team_stats"
        )

        return result_df

    def collect_standings(
        self,
        season: str = "2024-25",
        season_type: str = "Regular Season"
    ) -> pd.DataFrame:
        """Collect league standings."""
        try:
            endpoint = self._make_request(
                LeagueStandings,
                season=season,
                season_type=season_type
            )
            df = endpoint.get_data_frames()[0]

            self.storage.save_dataset(
                df,
                dataset_name=f"standings_{season}",
                description=f"League standings for {season}",
                season=season,
                data_type="standings"
            )

            return df

        except Exception as e:
            warnings.warn(f"Failed to get standings: {e}")
            return pd.DataFrame()

    # ===================
    # Game Data Collection
    # ===================

    def collect_league_game_logs(
        self,
        seasons: Optional[List[str]] = None,
        season_type: str = "Regular Season"
    ) -> pd.DataFrame:
        """
        Collect league-wide game logs.

        This is more efficient than collecting per-player for large-scale analysis.
        """
        if seasons is None:
            seasons = ["2024-25", "2023-24", "2022-23"]

        all_logs = []

        for season in seasons:
            try:
                # Player logs
                endpoint = self._make_request(
                    LeagueGameLog,
                    season=season,
                    season_type_all_star=season_type,
                    player_or_team_abbreviation="P"  # Players
                )
                df = endpoint.get_data_frames()[0]
                if len(df) > 0:
                    df['SEASON'] = season
                    all_logs.append(df)
                    print(f"Collected {len(df)} player game logs for {season}")

            except Exception as e:
                warnings.warn(f"Failed to get league game logs for {season}: {e}")

        if not all_logs:
            return pd.DataFrame()

        result_df = pd.concat(all_logs, ignore_index=True)

        self.storage.save_dataset(
            result_df,
            dataset_name="league_player_game_logs",
            description=f"League-wide player game logs for {seasons}",
            season=",".join(seasons),
            data_type="player_stats"
        )

        return result_df

    def collect_box_scores(
        self,
        game_ids: List[str],
        include_advanced: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Collect detailed box scores for games.

        Returns:
            Tuple of (traditional_box_scores, advanced_box_scores)
        """
        traditional_scores = []
        advanced_scores = []

        for i, game_id in enumerate(game_ids):
            try:
                # Traditional box score
                trad_endpoint = self._make_request(
                    BoxScoreTraditionalV2,
                    game_id=game_id
                )
                player_stats = trad_endpoint.player_stats.get_data_frame()
                if len(player_stats) > 0:
                    player_stats['GAME_ID'] = game_id
                    traditional_scores.append(player_stats)

                if include_advanced:
                    adv_endpoint = self._make_request(
                        BoxScoreAdvancedV2,
                        game_id=game_id
                    )
                    adv_stats = adv_endpoint.player_stats.get_data_frame()
                    if len(adv_stats) > 0:
                        adv_stats['GAME_ID'] = game_id
                        advanced_scores.append(adv_stats)

            except Exception as e:
                warnings.warn(f"Failed to get box score for game {game_id}: {e}")

            if (i + 1) % 20 == 0:
                print(f"Progress: {i + 1}/{len(game_ids)}")

        trad_df = pd.concat(traditional_scores, ignore_index=True) if traditional_scores else pd.DataFrame()
        adv_df = pd.concat(advanced_scores, ignore_index=True) if advanced_scores else pd.DataFrame()

        if len(trad_df) > 0:
            self.storage.save_dataset(
                trad_df,
                dataset_name="box_scores_traditional",
                description="Traditional box scores",
                data_type="box_scores"
            )

        if len(adv_df) > 0:
            self.storage.save_dataset(
                adv_df,
                dataset_name="box_scores_advanced",
                description="Advanced box scores",
                data_type="box_scores"
            )

        return trad_df, adv_df

    # ===================
    # Full Data Collection Pipeline
    # ===================

    def collect_all_data(
        self,
        seasons: Optional[List[str]] = None,
        include_box_scores: bool = False,
        max_players: Optional[int] = None
    ) -> dict:
        """
        Collect all data needed for statistical modeling.

        Args:
            seasons: List of seasons to collect
            include_box_scores: Whether to collect detailed box scores (slow)
            max_players: Limit number of players (for testing)

        Returns:
            Dictionary with collection statistics
        """
        if seasons is None:
            seasons = ["2024-25", "2023-24", "2022-23"]

        stats = {
            'start_time': datetime.now().isoformat(),
            'seasons': seasons,
        }

        print("=" * 50)
        print("NBA Data Collection Pipeline")
        print("=" * 50)

        # Collect teams
        print("\n[1/5] Collecting teams...")
        teams_df = self.collect_all_teams()
        stats['teams_collected'] = len(teams_df)
        print(f"  Collected {len(teams_df)} teams")

        # Collect players
        print("\n[2/5] Collecting players...")
        players_df = self.collect_all_players(active_only=True)
        stats['players_collected'] = len(players_df)
        print(f"  Collected {len(players_df)} players")

        # Collect league game logs (more efficient than per-player)
        print("\n[3/5] Collecting league game logs...")
        game_logs_df = self.collect_league_game_logs(seasons=seasons)
        stats['game_logs_collected'] = len(game_logs_df)
        print(f"  Collected {len(game_logs_df)} game log entries")

        # Collect team game logs
        print("\n[4/5] Collecting team game logs...")
        team_logs_df = self.collect_team_game_logs(seasons=seasons)
        stats['team_logs_collected'] = len(team_logs_df)
        print(f"  Collected {len(team_logs_df)} team game log entries")

        # Collect standings
        print("\n[5/5] Collecting standings...")
        for season in seasons:
            self.collect_standings(season=season)
        stats['standings_collected'] = len(seasons)
        print(f"  Collected standings for {len(seasons)} seasons")

        # Optional: Box scores (slow)
        if include_box_scores and len(game_logs_df) > 0:
            print("\n[Bonus] Collecting detailed box scores...")
            game_ids = game_logs_df['GAME_ID'].unique()[:100]  # Limit to 100 games
            self.collect_box_scores(list(game_ids))
            stats['box_scores_collected'] = len(game_ids)

        stats['end_time'] = datetime.now().isoformat()
        stats['storage_stats'] = self.storage.get_storage_stats()

        print("\n" + "=" * 50)
        print("Collection Complete!")
        print("=" * 50)
        print(f"Storage stats: {stats['storage_stats']}")

        return stats
