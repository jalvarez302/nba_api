"""
Feature Engineering Module for NBA Statistics Modeling

Provides comprehensive feature engineering for basketball statistics prediction,
including rolling averages, rest days, home/away splits, and opponent strength.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class FeatureEngineer:
    """
    Feature engineering for NBA statistics modeling.

    Creates features commonly used in sports analytics:
    - Rolling averages (various windows)
    - Momentum/trend features
    - Rest days
    - Home/away indicators
    - Opponent strength metrics
    - Seasonal patterns
    """

    # Default rolling windows for averaging
    DEFAULT_WINDOWS = [3, 5, 10, 20]

    # Stats to compute rolling averages for
    CORE_STATS = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'MIN',
                  'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA']

    def __init__(
        self,
        rolling_windows: Optional[List[int]] = None,
        include_advanced: bool = True
    ):
        """
        Initialize the feature engineer.

        Args:
            rolling_windows: List of window sizes for rolling averages
            include_advanced: Whether to compute advanced metrics
        """
        self.rolling_windows = rolling_windows or self.DEFAULT_WINDOWS
        self.include_advanced = include_advanced

    def engineer_player_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'PTS'
    ) -> pd.DataFrame:
        """
        Engineer features for player-level prediction.

        Args:
            df: DataFrame with player game logs (must have PLAYER_ID, GAME_DATE)
            target_col: Target column for prediction

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()

        # Ensure proper types
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])

        # Numeric conversions
        for col in self.CORE_STATS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Group by player for rolling calculations
        player_groups = df.groupby('PLAYER_ID')

        # Rolling averages for each stat
        for stat in self.CORE_STATS:
            if stat not in df.columns:
                continue

            for window in self.rolling_windows:
                # Shift by 1 to avoid data leakage (use only past games)
                df[f'{stat}_avg_{window}'] = player_groups[stat].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )

                # Standard deviation for variance
                df[f'{stat}_std_{window}'] = player_groups[stat].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).std()
                )

        # Momentum features (trend over last N games)
        for stat in ['PTS', 'MIN']:
            if stat in df.columns:
                # Slope of recent performance (simplified momentum)
                df[f'{stat}_momentum'] = player_groups[stat].transform(
                    lambda x: x.shift(1).rolling(5, min_periods=3).apply(
                        lambda v: np.polyfit(range(len(v)), v, 1)[0] if len(v) >= 3 else 0
                    )
                )

        # Days rest calculation
        df['DAYS_REST'] = player_groups['GAME_DATE'].transform(
            lambda x: x.diff().dt.days.fillna(3)
        )
        df['DAYS_REST'] = df['DAYS_REST'].clip(upper=7)  # Cap at 7 days

        # Back-to-back indicator
        df['IS_BACK_TO_BACK'] = (df['DAYS_REST'] == 1).astype(int)

        # Home/Away indicator
        if 'MATCHUP' in df.columns:
            df['IS_HOME'] = (~df['MATCHUP'].str.contains('@', na=False)).astype(int)
        elif 'HOME_TEAM_ID' in df.columns and 'TEAM_ID' in df.columns:
            df['IS_HOME'] = (df['HOME_TEAM_ID'] == df['TEAM_ID']).astype(int)
        else:
            df['IS_HOME'] = 0

        # Games played this season (cumulative)
        df['GAMES_PLAYED_SEASON'] = player_groups.cumcount() + 1

        if self.include_advanced:
            df = self._add_advanced_features(df)

        # Calculate shooting percentages (rolling)
        df = self._add_shooting_percentages(df)

        return df

    def _add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced analytics features."""

        # Usage rate proxy (if minutes available)
        if 'MIN' in df.columns and 'FGA' in df.columns:
            df['USAGE_PROXY'] = df['FGA'] / (df['MIN'] + 0.1)

        # True Shooting Attempts
        if all(col in df.columns for col in ['FGA', 'FTA']):
            df['TSA'] = df['FGA'] + 0.44 * df['FTA']

        # Points per TSA (efficiency)
        if 'TSA' in df.columns and 'PTS' in df.columns:
            df['PTS_PER_TSA'] = df['PTS'] / (df['TSA'] + 0.1)

        # Fantasy points (simplified DraftKings scoring)
        if all(col in df.columns for col in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']):
            df['FANTASY_PTS'] = (
                df['PTS'] +
                df['REB'] * 1.25 +
                df['AST'] * 1.5 +
                df['STL'] * 2 +
                df['BLK'] * 2 -
                df['TOV'] * 0.5
            )

        return df

    def _add_shooting_percentages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling shooting percentage features."""
        player_groups = df.groupby('PLAYER_ID')

        # Field Goal Percentage
        if all(col in df.columns for col in ['FGM', 'FGA']):
            df['FGM_rolling'] = player_groups['FGM'].transform(
                lambda x: x.shift(1).rolling(10, min_periods=3).sum()
            )
            df['FGA_rolling'] = player_groups['FGA'].transform(
                lambda x: x.shift(1).rolling(10, min_periods=3).sum()
            )
            df['FG_PCT_rolling'] = df['FGM_rolling'] / (df['FGA_rolling'] + 0.1)

        # Three-Point Percentage
        if all(col in df.columns for col in ['FG3M', 'FG3A']):
            df['FG3M_rolling'] = player_groups['FG3M'].transform(
                lambda x: x.shift(1).rolling(10, min_periods=3).sum()
            )
            df['FG3A_rolling'] = player_groups['FG3A'].transform(
                lambda x: x.shift(1).rolling(10, min_periods=3).sum()
            )
            df['FG3_PCT_rolling'] = df['FG3M_rolling'] / (df['FG3A_rolling'] + 0.1)

        # Free Throw Percentage
        if all(col in df.columns for col in ['FTM', 'FTA']):
            df['FTM_rolling'] = player_groups['FTM'].transform(
                lambda x: x.shift(1).rolling(10, min_periods=3).sum()
            )
            df['FTA_rolling'] = player_groups['FTA'].transform(
                lambda x: x.shift(1).rolling(10, min_periods=3).sum()
            )
            df['FT_PCT_rolling'] = df['FTM_rolling'] / (df['FTA_rolling'] + 0.1)

        return df

    def engineer_game_features(
        self,
        games_df: pd.DataFrame,
        team_stats_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Engineer features for game outcome prediction.

        Args:
            games_df: DataFrame with game data
            team_stats_df: Optional team statistics for enrichment

        Returns:
            DataFrame with engineered features for game prediction
        """
        df = games_df.copy()

        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df = df.sort_values('GAME_DATE')

        # If we have team stats, compute rolling team metrics
        if team_stats_df is not None:
            df = self._add_team_strength_features(df, team_stats_df)

        # Win streak calculations
        if 'WL' in df.columns and 'TEAM_ID' in df.columns:
            df = self._add_streak_features(df)

        return df

    def _add_team_strength_features(
        self,
        games_df: pd.DataFrame,
        team_stats_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add team strength metrics based on historical performance."""

        # Calculate team averages
        if 'PTS' in team_stats_df.columns and 'TEAM_ID' in team_stats_df.columns:
            team_avgs = team_stats_df.groupby('TEAM_ID').agg({
                'PTS': 'mean',
            }).rename(columns={'PTS': 'TEAM_AVG_PTS'})

            if 'HOME_TEAM_ID' in games_df.columns:
                games_df = games_df.merge(
                    team_avgs,
                    left_on='HOME_TEAM_ID',
                    right_index=True,
                    how='left',
                    suffixes=('', '_home')
                )

            if 'AWAY_TEAM_ID' in games_df.columns:
                games_df = games_df.merge(
                    team_avgs,
                    left_on='AWAY_TEAM_ID',
                    right_index=True,
                    how='left',
                    suffixes=('', '_away')
                )

        return games_df

    def _add_streak_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add win/loss streak features."""

        def calculate_streak(wl_series):
            """Calculate current win/loss streak."""
            streaks = []
            current_streak = 0
            current_type = None

            for result in wl_series:
                if pd.isna(result):
                    streaks.append(0)
                    continue

                if result == current_type:
                    current_streak += 1 if result == 'W' else -1
                else:
                    current_streak = 1 if result == 'W' else -1
                    current_type = result

                streaks.append(current_streak)

            return streaks

        df['WIN_STREAK'] = df.groupby('TEAM_ID')['WL'].transform(calculate_streak)

        return df

    def get_feature_columns(self, df: pd.DataFrame, target_col: str) -> List[str]:
        """
        Get list of feature columns suitable for modeling.

        Excludes target, ID columns, and raw stats that would cause leakage.
        """
        exclude_patterns = [
            target_col, 'PLAYER_ID', 'TEAM_ID', 'GAME_ID', 'GAME_DATE',
            'MATCHUP', 'WL', 'VIDEO_AVAILABLE', 'SEASON', 'SEASON_ID',
            'Player_ID', 'Game_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION',
            'TEAM_NAME', 'MIN',  # Exclude raw MIN to avoid leakage
        ]

        # Also exclude raw stats (use only rolling versions)
        exclude_patterns.extend(self.CORE_STATS)

        feature_cols = []
        for col in df.columns:
            # Skip excluded columns
            if any(pattern == col or col.startswith(pattern + '_')
                   for pattern in ['PLAYER_ID', 'TEAM_ID', 'GAME_ID', 'GAME_DATE', 'MATCHUP']):
                continue

            if col in exclude_patterns:
                continue

            # Only include numeric columns
            if df[col].dtype in [np.int64, np.float64, np.int32, np.float32]:
                feature_cols.append(col)

        return feature_cols

    def prepare_training_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        dropna: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for model training.

        Args:
            df: DataFrame with engineered features
            target_col: Name of target column
            feature_cols: List of feature columns (auto-detected if None)
            dropna: Whether to drop rows with missing values

        Returns:
            Tuple of (X, y) for training
        """
        if feature_cols is None:
            feature_cols = self.get_feature_columns(df, target_col)

        X = df[feature_cols].copy()
        y = df[target_col].copy()

        if dropna:
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]

        return X, y
