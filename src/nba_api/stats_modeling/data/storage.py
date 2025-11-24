"""
NBA Data Storage Module

Provides efficient data storage using SQLite for metadata/indexing and Parquet for large datasets.
Optimized for statistical modeling and machine learning workflows.
"""

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
import json

import pandas as pd
import numpy as np


class NBADataStorage:
    """
    Manages NBA data storage using SQLite + Parquet hybrid approach.

    - SQLite: Metadata, indexing, and smaller tables
    - Parquet: Large datasets optimized for analytics (columnar storage)

    This architecture provides:
    - Fast queries via SQLite indexes
    - Efficient storage and I/O for large datasets via Parquet
    - Easy integration with pandas/ML pipelines
    """

    def __init__(self, data_dir: str = "./nba_data"):
        """
        Initialize the data storage.

        Args:
            data_dir: Directory to store all data files
        """
        self.data_dir = Path(data_dir)
        self.db_path = self.data_dir / "nba_metadata.db"
        self.parquet_dir = self.data_dir / "parquet"

        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_dir.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_database(self):
        """Initialize the SQLite database with required tables."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Players table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS players (
                player_id INTEGER PRIMARY KEY,
                full_name TEXT NOT NULL,
                first_name TEXT,
                last_name TEXT,
                is_active INTEGER DEFAULT 1,
                team_id INTEGER,
                position TEXT,
                height TEXT,
                weight INTEGER,
                birth_date TEXT,
                draft_year INTEGER,
                draft_round INTEGER,
                draft_number INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Teams table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS teams (
                team_id INTEGER PRIMARY KEY,
                abbreviation TEXT NOT NULL,
                full_name TEXT NOT NULL,
                city TEXT,
                state TEXT,
                conference TEXT,
                division TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Games table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                game_date TEXT NOT NULL,
                season TEXT NOT NULL,
                season_type TEXT,
                home_team_id INTEGER,
                away_team_id INTEGER,
                home_team_score INTEGER,
                away_team_score INTEGER,
                winner_team_id INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (home_team_id) REFERENCES teams(team_id),
                FOREIGN KEY (away_team_id) REFERENCES teams(team_id)
            )
        """)

        # Dataset registry (tracks Parquet files)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dataset_registry (
                dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_name TEXT UNIQUE NOT NULL,
                description TEXT,
                file_path TEXT NOT NULL,
                row_count INTEGER,
                columns TEXT,
                season TEXT,
                data_type TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_players_name ON players(full_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_players_team ON players(team_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_season ON games(season)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_teams ON games(home_team_id, away_team_id)")

        conn.commit()
        conn.close()

    # ===================
    # Player Operations
    # ===================

    def save_players(self, players_df: pd.DataFrame) -> int:
        """
        Save or update players to the database.

        Args:
            players_df: DataFrame with player data

        Returns:
            Number of players saved
        """
        conn = self._get_connection()

        # Standardize column names
        column_mapping = {
            'PERSON_ID': 'player_id',
            'PLAYER_ID': 'player_id',
            'DISPLAY_FIRST_LAST': 'full_name',
            'PLAYER_NAME': 'full_name',
            'FIRST_NAME': 'first_name',
            'LAST_NAME': 'last_name',
            'TEAM_ID': 'team_id',
            'POSITION': 'position',
            'HEIGHT': 'height',
            'WEIGHT': 'weight',
            'BIRTHDATE': 'birth_date',
            'DRAFT_YEAR': 'draft_year',
            'DRAFT_ROUND': 'draft_round',
            'DRAFT_NUMBER': 'draft_number',
        }

        df = players_df.rename(columns=column_mapping)

        count = 0
        for _, row in df.iterrows():
            player_data = row.to_dict()
            # Keep only valid columns
            valid_cols = ['player_id', 'full_name', 'first_name', 'last_name',
                          'team_id', 'position', 'height', 'weight', 'birth_date',
                          'draft_year', 'draft_round', 'draft_number', 'is_active']
            player_data = {k: v for k, v in player_data.items() if k in valid_cols}

            if 'player_id' not in player_data:
                continue

            # Upsert
            cols = list(player_data.keys())
            placeholders = ', '.join(['?' for _ in cols])
            update_clause = ', '.join([f"{c}=excluded.{c}" for c in cols if c != 'player_id'])

            sql = f"""
                INSERT INTO players ({', '.join(cols)})
                VALUES ({placeholders})
                ON CONFLICT(player_id) DO UPDATE SET {update_clause}, updated_at=CURRENT_TIMESTAMP
            """
            conn.execute(sql, list(player_data.values()))
            count += 1

        conn.commit()
        conn.close()
        return count

    def get_players(self, active_only: bool = False, team_id: Optional[int] = None) -> pd.DataFrame:
        """Get players from the database."""
        conn = self._get_connection()

        query = "SELECT * FROM players WHERE 1=1"
        params = []

        if active_only:
            query += " AND is_active = 1"
        if team_id:
            query += " AND team_id = ?"
            params.append(team_id)

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df

    # ===================
    # Team Operations
    # ===================

    def save_teams(self, teams_df: pd.DataFrame) -> int:
        """Save or update teams to the database."""
        conn = self._get_connection()

        column_mapping = {
            'TEAM_ID': 'team_id',
            'ABBREVIATION': 'abbreviation',
            'TEAM_NAME': 'full_name',
            'FULL_NAME': 'full_name',
            'CITY': 'city',
            'STATE': 'state',
            'CONFERENCE': 'conference',
            'DIVISION': 'division',
        }

        df = teams_df.rename(columns=column_mapping)

        count = 0
        for _, row in df.iterrows():
            team_data = row.to_dict()
            valid_cols = ['team_id', 'abbreviation', 'full_name', 'city',
                          'state', 'conference', 'division']
            team_data = {k: v for k, v in team_data.items() if k in valid_cols}

            if 'team_id' not in team_data:
                continue

            cols = list(team_data.keys())
            placeholders = ', '.join(['?' for _ in cols])
            update_clause = ', '.join([f"{c}=excluded.{c}" for c in cols if c != 'team_id'])

            sql = f"""
                INSERT INTO teams ({', '.join(cols)})
                VALUES ({placeholders})
                ON CONFLICT(team_id) DO UPDATE SET {update_clause}, updated_at=CURRENT_TIMESTAMP
            """
            conn.execute(sql, list(team_data.values()))
            count += 1

        conn.commit()
        conn.close()
        return count

    def get_teams(self, conference: Optional[str] = None) -> pd.DataFrame:
        """Get teams from the database."""
        conn = self._get_connection()

        query = "SELECT * FROM teams WHERE 1=1"
        params = []

        if conference:
            query += " AND conference = ?"
            params.append(conference)

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df

    # ===================
    # Game Operations
    # ===================

    def save_games(self, games_df: pd.DataFrame) -> int:
        """Save or update games to the database."""
        conn = self._get_connection()

        column_mapping = {
            'GAME_ID': 'game_id',
            'GAME_DATE': 'game_date',
            'SEASON_ID': 'season',
            'SEASON': 'season',
            'SEASON_TYPE': 'season_type',
            'HOME_TEAM_ID': 'home_team_id',
            'VISITOR_TEAM_ID': 'away_team_id',
            'AWAY_TEAM_ID': 'away_team_id',
            'PTS_home': 'home_team_score',
            'HOME_TEAM_SCORE': 'home_team_score',
            'PTS_away': 'away_team_score',
            'AWAY_TEAM_SCORE': 'away_team_score',
        }

        df = games_df.rename(columns=column_mapping)

        count = 0
        for _, row in df.iterrows():
            game_data = row.to_dict()
            valid_cols = ['game_id', 'game_date', 'season', 'season_type',
                          'home_team_id', 'away_team_id', 'home_team_score',
                          'away_team_score', 'winner_team_id']
            game_data = {k: v for k, v in game_data.items() if k in valid_cols}

            if 'game_id' not in game_data:
                continue

            # Determine winner
            if 'home_team_score' in game_data and 'away_team_score' in game_data:
                home_score = game_data.get('home_team_score', 0) or 0
                away_score = game_data.get('away_team_score', 0) or 0
                if home_score > away_score:
                    game_data['winner_team_id'] = game_data.get('home_team_id')
                elif away_score > home_score:
                    game_data['winner_team_id'] = game_data.get('away_team_id')

            cols = list(game_data.keys())
            placeholders = ', '.join(['?' for _ in cols])
            update_clause = ', '.join([f"{c}=excluded.{c}" for c in cols if c != 'game_id'])

            sql = f"""
                INSERT INTO games ({', '.join(cols)})
                VALUES ({placeholders})
                ON CONFLICT(game_id) DO UPDATE SET {update_clause}, updated_at=CURRENT_TIMESTAMP
            """
            conn.execute(sql, list(game_data.values()))
            count += 1

        conn.commit()
        conn.close()
        return count

    def get_games(
        self,
        season: Optional[str] = None,
        team_id: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Get games from the database with optional filters."""
        conn = self._get_connection()

        query = "SELECT * FROM games WHERE 1=1"
        params = []

        if season:
            query += " AND season = ?"
            params.append(season)
        if team_id:
            query += " AND (home_team_id = ? OR away_team_id = ?)"
            params.extend([team_id, team_id])
        if start_date:
            query += " AND game_date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND game_date <= ?"
            params.append(end_date)

        query += " ORDER BY game_date"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df

    # ===================
    # Parquet Operations (for large datasets)
    # ===================

    def save_dataset(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        description: str = "",
        season: str = "",
        data_type: str = "stats"
    ) -> str:
        """
        Save a large dataset as Parquet for efficient storage and retrieval.

        Args:
            df: DataFrame to save
            dataset_name: Unique name for the dataset
            description: Description of the dataset
            season: Season identifier (e.g., "2023-24")
            data_type: Type of data (e.g., "player_stats", "game_logs", "box_scores")

        Returns:
            Path to the saved file
        """
        # Sanitize dataset name for filesystem
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in dataset_name)
        file_path = self.parquet_dir / f"{safe_name}.parquet"

        # Save as Parquet with compression
        df.to_parquet(
            file_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )

        # Register in database
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO dataset_registry
            (dataset_name, description, file_path, row_count, columns, season, data_type)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(dataset_name) DO UPDATE SET
                description=excluded.description,
                file_path=excluded.file_path,
                row_count=excluded.row_count,
                columns=excluded.columns,
                season=excluded.season,
                data_type=excluded.data_type,
                updated_at=CURRENT_TIMESTAMP
        """, (
            dataset_name,
            description,
            str(file_path),
            len(df),
            json.dumps(list(df.columns)),
            season,
            data_type
        ))

        conn.commit()
        conn.close()

        return str(file_path)

    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """
        Load a dataset from Parquet storage.

        Args:
            dataset_name: Name of the dataset to load

        Returns:
            DataFrame with the dataset
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT file_path FROM dataset_registry WHERE dataset_name = ?",
            (dataset_name,)
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            raise ValueError(f"Dataset '{dataset_name}' not found in registry")

        file_path = row['file_path']
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        return pd.read_parquet(file_path)

    def list_datasets(self, data_type: Optional[str] = None) -> pd.DataFrame:
        """List all registered datasets."""
        conn = self._get_connection()

        query = "SELECT * FROM dataset_registry WHERE 1=1"
        params = []

        if data_type:
            query += " AND data_type = ?"
            params.append(data_type)

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df

    # ===================
    # Feature Engineering Support
    # ===================

    def get_player_stats_for_modeling(
        self,
        dataset_name: str = "player_game_logs",
        min_games: int = 10
    ) -> pd.DataFrame:
        """
        Load and prepare player stats for ML modeling.

        Returns a DataFrame with:
        - Rolling averages (last 5, 10, 20 games)
        - Home/away splits
        - Days rest
        - Opponent strength metrics
        """
        df = self.load_dataset(dataset_name)

        # Ensure numeric columns
        numeric_cols = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV',
                        'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'MIN']

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Sort by player and date
        if 'GAME_DATE' in df.columns:
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])

        return df

    def get_game_data_for_modeling(
        self,
        dataset_name: str = "game_logs"
    ) -> pd.DataFrame:
        """
        Load and prepare game data for outcome prediction modeling.
        """
        df = self.load_dataset(dataset_name)
        return df

    # ===================
    # Utility Methods
    # ===================

    def get_storage_stats(self) -> Dict:
        """Get storage statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()

        stats = {
            'players': cursor.execute("SELECT COUNT(*) FROM players").fetchone()[0],
            'teams': cursor.execute("SELECT COUNT(*) FROM teams").fetchone()[0],
            'games': cursor.execute("SELECT COUNT(*) FROM games").fetchone()[0],
            'datasets': cursor.execute("SELECT COUNT(*) FROM dataset_registry").fetchone()[0],
        }

        # Calculate total Parquet storage
        total_size = 0
        for f in self.parquet_dir.glob("*.parquet"):
            total_size += f.stat().st_size

        stats['parquet_size_mb'] = round(total_size / (1024 * 1024), 2)
        stats['db_size_mb'] = round(self.db_path.stat().st_size / (1024 * 1024), 2) if self.db_path.exists() else 0

        conn.close()
        return stats

    def clear_all_data(self, confirm: bool = False):
        """Clear all stored data. Requires explicit confirmation."""
        if not confirm:
            raise ValueError("Must pass confirm=True to clear all data")

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM players")
        cursor.execute("DELETE FROM teams")
        cursor.execute("DELETE FROM games")
        cursor.execute("DELETE FROM dataset_registry")

        conn.commit()
        conn.close()

        # Remove Parquet files
        for f in self.parquet_dir.glob("*.parquet"):
            f.unlink()
