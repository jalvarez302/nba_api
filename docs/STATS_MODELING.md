# NBA Statistics Modeling Package

A comprehensive machine learning package for predicting NBA player and game statistics using modern best practices.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Data Collection](#data-collection)
4. [Data Storage](#data-storage)
5. [Feature Engineering](#feature-engineering)
6. [Prediction Models](#prediction-models)
7. [API Reference](#api-reference)
8. [Examples](#examples)
9. [Best Practices](#best-practices)

---

## Installation

### Basic Installation (sklearn backend)

```bash
pip install nba_api[modeling]
```

### Advanced Installation (with LightGBM and XGBoost)

```bash
pip install nba_api[modeling-advanced]
```

### Dependencies

| Package | Basic | Advanced | Purpose |
|---------|-------|----------|---------|
| scikit-learn | ✓ | ✓ | ML models, preprocessing |
| pyarrow | ✓ | ✓ | Parquet file support |
| scipy | ✓ | ✓ | Statistical functions |
| lightgbm | - | ✓ | Fast gradient boosting |
| xgboost | - | ✓ | Extreme gradient boosting |

---

## Quick Start

### 5-Minute Example

```python
from nba_api.stats_modeling import NBAPredictor

# Initialize the predictor
predictor = NBAPredictor()

# Step 1: Collect data (takes 10-30 minutes for full dataset)
predictor.collect_data(seasons=['2024-25', '2023-24'])

# Step 2: Train all models
predictor.train_all_models()

# Step 3: Make predictions
# Predict points for LeBron James
lebron_id = predictor.get_player_id("LeBron James")
points_pred = predictor.predict_points(lebron_id)
print(f"Predicted points: {points_pred['predicted_points']:.1f}")

# Predict game outcome: Lakers vs Warriors
lakers_id = predictor.get_team_id("Lakers")
warriors_id = predictor.get_team_id("Warriors")
game_pred = predictor.predict_game(lakers_id, warriors_id)
print(f"Lakers win probability: {game_pred['home_win_probability']:.1%}")

# Save models for later use
predictor.save_all_models()
```

### One-Line Setup

```python
from nba_api.stats_modeling import create_nba_predictor

# Collect data and train models in one step
predictor = create_nba_predictor(
    collect_seasons=['2024-25', '2023-24'],
    train_models=True
)
```

---

## Data Collection

### NBADataCollector

The `NBADataCollector` class handles fetching data from the NBA API with rate limiting and retry logic.

```python
from nba_api.stats_modeling import NBADataCollector, NBADataStorage

# Initialize
storage = NBADataStorage("./nba_data")
collector = NBADataCollector(storage=storage)

# Collect all data for specified seasons
stats = collector.collect_all_data(
    seasons=['2024-25', '2023-24', '2022-23'],
    include_box_scores=False  # Set True for detailed stats (slower)
)
```

### Available Collection Methods

| Method | Description | Speed |
|--------|-------------|-------|
| `collect_all_players()` | All NBA/WNBA players | Fast |
| `collect_all_teams()` | All NBA teams | Fast |
| `collect_player_game_logs()` | Individual player game logs | Medium |
| `collect_team_game_logs()` | Team game logs | Medium |
| `collect_league_game_logs()` | League-wide player stats | Medium |
| `collect_standings()` | League standings | Fast |
| `collect_box_scores()` | Detailed box scores | Slow |
| `collect_all_data()` | Complete pipeline | 10-30 min |

### Rate Limiting

The collector automatically handles rate limiting:

```python
collector = NBADataCollector(
    request_delay=0.6  # Seconds between requests (default)
)
```

---

## Data Storage

### NBADataStorage

Hybrid SQLite + Parquet storage optimized for machine learning.

```python
from nba_api.stats_modeling import NBADataStorage

storage = NBADataStorage("./nba_data")
```

### Storage Architecture

```
nba_data/
├── nba_metadata.db          # SQLite: players, teams, games, registry
└── parquet/
    ├── player_game_logs.parquet
    ├── team_game_logs.parquet
    └── standings_2024-25.parquet
```

### SQLite Tables

| Table | Purpose |
|-------|---------|
| `players` | Player metadata (ID, name, team, position) |
| `teams` | Team metadata (ID, name, conference) |
| `games` | Game results and scores |
| `dataset_registry` | Tracks Parquet files |

### Working with Storage

```python
# Save players
storage.save_players(players_df)

# Get active players
active_players = storage.get_players(active_only=True)

# Get games for a team
lakers_games = storage.get_games(
    team_id=1610612747,
    season='2024-25'
)

# Save large dataset as Parquet
storage.save_dataset(
    df=game_logs_df,
    dataset_name="player_game_logs",
    description="Player game logs for 2024-25",
    season="2024-25",
    data_type="player_stats"
)

# Load dataset
df = storage.load_dataset("player_game_logs")

# List all datasets
datasets = storage.list_datasets()

# Get storage statistics
stats = storage.get_storage_stats()
# Returns: {'players': 500, 'teams': 30, 'games': 1000,
#           'datasets': 5, 'parquet_size_mb': 25.4}
```

---

## Feature Engineering

### FeatureEngineer

Automatically creates ML-ready features from raw game logs.

```python
from nba_api.stats_modeling.utils import FeatureEngineer

fe = FeatureEngineer(
    rolling_windows=[3, 5, 10, 20],  # Games to average over
    include_advanced=True             # Include advanced metrics
)

# Engineer features for player prediction
engineered_df = fe.engineer_player_features(
    df=game_logs_df,
    target_col='PTS'
)
```

### Features Created

#### Rolling Averages
For each stat (PTS, REB, AST, STL, BLK, TOV, FGM, FGA, FG3M, FG3A, FTM, FTA, MIN):
- `{STAT}_avg_3` - Last 3 games average
- `{STAT}_avg_5` - Last 5 games average
- `{STAT}_avg_10` - Last 10 games average
- `{STAT}_avg_20` - Last 20 games average
- `{STAT}_std_5` - Standard deviation (variance indicator)

#### Momentum Features
- `PTS_momentum` - Trend slope over last 5 games
- `MIN_momentum` - Playing time trend

#### Contextual Features
- `DAYS_REST` - Days since last game (capped at 7)
- `IS_BACK_TO_BACK` - 1 if back-to-back game
- `IS_HOME` - 1 if home game
- `GAMES_PLAYED_SEASON` - Games played this season

#### Shooting Percentages (Rolling)
- `FG_PCT_rolling` - Field goal % (last 10 games)
- `FG3_PCT_rolling` - Three-point % (last 10 games)
- `FT_PCT_rolling` - Free throw % (last 10 games)

#### Advanced Metrics
- `USAGE_PROXY` - Estimated usage rate
- `TSA` - True shooting attempts
- `PTS_PER_TSA` - Points per TSA (efficiency)
- `FANTASY_PTS` - DraftKings-style fantasy points

### Preparing Training Data

```python
# Get feature columns (automatically excludes leaky features)
feature_cols = fe.get_feature_columns(engineered_df, target_col='PTS')

# Prepare X and y for training
X, y = fe.prepare_training_data(
    df=engineered_df,
    target_col='PTS',
    feature_cols=feature_cols,
    dropna=True
)
```

---

## Prediction Models

### 1. PointsPredictor

Predicts player points scored.

```python
from nba_api.stats_modeling import PointsPredictor

# Initialize
model = PointsPredictor(
    backend='sklearn',  # or 'lightgbm', 'xgboost'
    model_name='points_model'
)

# Train
metrics = model.fit(
    df=game_logs_df,
    target_col='PTS',
    validation_split=0.2
)
print(f"Validation RMSE: {metrics['validation_metrics']['rmse']:.2f}")

# Predict
predictions = model.predict(new_data_df)

# Predict with confidence intervals
pred, lower, upper = model.predict_with_confidence(
    new_data_df,
    confidence_level=0.9
)

# Predict for specific player
result = model.predict_player(
    player_id=2544,  # LeBron
    game_logs_df=all_logs_df,
    n_games_context=20
)
# Returns: {'predicted_points': 25.3, 'confidence_lower': 18.1,
#           'confidence_upper': 32.5, 'recent_average': 24.8}

# Cross-validate
cv_results = model.cross_validate(
    df=game_logs_df,
    target_col='PTS',
    n_splits=5
)

# Feature importance
importance = model.get_feature_importance(top_n=20)

# Save/load model
model.save('points_model.pkl')
model.load('points_model.pkl')
```

### 2. GameOutcomePredictor

Predicts game outcomes (win/loss probability).

```python
from nba_api.stats_modeling import GameOutcomePredictor

model = GameOutcomePredictor(
    backend='sklearn',
    prediction_type='win_probability'  # or 'point_spread', 'total_points'
)

# Train on team game logs
metrics = model.fit(
    df=team_logs_df,
    target_col='WIN'
)
print(f"Validation Accuracy: {metrics['validation_metrics']['accuracy']:.1%}")

# Predict win probability
win_probs = model.predict_proba(team_data_df)

# Predict specific matchup
result = model.predict_game(
    home_team_logs=lakers_logs,
    away_team_logs=warriors_logs,
    n_games_context=10
)
# Returns: {'home_win_probability': 0.58, 'away_win_probability': 0.42,
#           'predicted_winner': 'home', 'confidence': 0.16}
```

### 3. IndividualStatsPredictor

Predicts multiple statistics simultaneously.

```python
from nba_api.stats_modeling import IndividualStatsPredictor

model = IndividualStatsPredictor(
    targets=['FG3M', 'FTM', 'BLK', 'FGM'],  # Stats to predict
    backend='sklearn'
)

# Train
metrics = model.fit(
    df=game_logs_df,
    strategy='individual'  # or 'multi' for multi-output
)

# Predict all stats
predictions = model.predict(new_data_df)
# Returns DataFrame with columns: FG3M_pred, FTM_pred, BLK_pred, FGM_pred

# Predict for specific player
result = model.predict_player_stats(
    player_id=201939,  # Stephen Curry
    game_logs_df=all_logs_df
)
# Returns: {'predictions': {'FG3M': 4.2, 'FTM': 3.1, 'BLK': 0.2, 'FGM': 9.8},
#           'recent_averages': {'FG3M': 4.5, 'FTM': 3.0, 'BLK': 0.3, 'FGM': 10.1}}
```

### Specialized Predictors

```python
from nba_api.stats_modeling import (
    ThreePointPredictor,   # Predicts FG3M
    FreeThrowPredictor,    # Predicts FTM
    BlocksPredictor,       # Predicts BLK
    FieldGoalPredictor     # Predicts FGM
)

# Each works the same way
three_pt_model = ThreePointPredictor()
three_pt_model.fit(game_logs_df)
predictions = three_pt_model.predict(new_data_df)
```

---

## API Reference

### NBAPredictor

The main high-level interface.

```python
from nba_api.stats_modeling import NBAPredictor

predictor = NBAPredictor(
    data_dir="./nba_data",     # Data storage directory
    model_dir="./nba_models",  # Model storage directory
    backend="sklearn"          # ML backend
)
```

#### Methods

| Method | Description |
|--------|-------------|
| `collect_data(seasons, include_box_scores)` | Collect NBA data |
| `load_data()` | Load collected data into memory |
| `train_points_model(**kwargs)` | Train points predictor |
| `train_game_outcome_model(**kwargs)` | Train game predictor |
| `train_individual_stats_model(targets, **kwargs)` | Train stats predictor |
| `train_all_models(**kwargs)` | Train all models |
| `predict_points(player_id, n_games_context)` | Predict player points |
| `predict_game(home_team_id, away_team_id)` | Predict game outcome |
| `predict_all_stats(player_id)` | Predict all player stats |
| `predict_three_pointers(player_id)` | Predict 3PM |
| `predict_free_throws(player_id)` | Predict FTM |
| `predict_blocks(player_id)` | Predict BLK |
| `predict_field_goals(player_id)` | Predict FGM |
| `get_player_id(name)` | Look up player ID |
| `get_team_id(name)` | Look up team ID |
| `save_all_models()` | Save trained models |
| `load_all_models()` | Load saved models |
| `get_storage_stats()` | Get storage statistics |
| `list_datasets()` | List available datasets |

---

## Examples

### Example 1: Daily Player Predictions

```python
from nba_api.stats_modeling import NBAPredictor

# Load pre-trained models
predictor = NBAPredictor()
predictor.load_all_models()
predictor.load_data()

# Players to predict
players = ["LeBron James", "Stephen Curry", "Giannis Antetokounmpo",
           "Luka Doncic", "Nikola Jokic"]

print("Today's Predictions")
print("=" * 50)

for name in players:
    player_id = predictor.get_player_id(name)
    if player_id:
        try:
            pts = predictor.predict_points(player_id)
            stats = predictor.predict_all_stats(player_id)

            print(f"\n{name}:")
            print(f"  Points: {pts['predicted_points']:.1f} "
                  f"(range: {pts['confidence_lower']:.1f}-{pts['confidence_upper']:.1f})")

            for stat, pred in stats['predictions'].items():
                avg = stats['recent_averages'].get(stat, 0)
                print(f"  {stat}: {pred:.1f} (avg: {avg:.1f})")
        except Exception as e:
            print(f"  Could not predict: {e}")
```

### Example 2: Game Predictions

```python
from nba_api.stats_modeling import NBAPredictor

predictor = NBAPredictor()
predictor.load_all_models()
predictor.load_data()

# Tonight's games
matchups = [
    ("Lakers", "Warriors"),
    ("Celtics", "Heat"),
    ("Nuggets", "Suns"),
]

print("Tonight's Game Predictions")
print("=" * 50)

for home, away in matchups:
    home_id = predictor.get_team_id(home)
    away_id = predictor.get_team_id(away)

    if home_id and away_id:
        pred = predictor.predict_game(home_id, away_id)

        print(f"\n{home} vs {away}")
        print(f"  {home} (home): {pred['home_win_probability']:.1%}")
        print(f"  {away} (away): {pred['away_win_probability']:.1%}")
        print(f"  Prediction: {home if pred['predicted_winner'] == 'home' else away}")
        print(f"  Confidence: {pred['confidence']:.1%}")
```

### Example 3: Model Comparison

```python
from nba_api.stats_modeling import PointsPredictor, NBADataStorage

storage = NBADataStorage("./nba_data")
df = storage.load_dataset("league_player_game_logs")

backends = ['sklearn']

# Add optional backends if available
try:
    import lightgbm
    backends.append('lightgbm')
except ImportError:
    pass

try:
    import xgboost
    backends.append('xgboost')
except ImportError:
    pass

print("Model Comparison")
print("=" * 50)

for backend in backends:
    model = PointsPredictor(backend=backend)
    cv_results = model.cross_validate(df, target_col='PTS', n_splits=5)

    print(f"\n{backend.upper()}:")
    print(f"  RMSE: {cv_results['rmse_mean']:.3f} (+/- {cv_results['rmse_std']:.3f})")
    print(f"  MAE:  {cv_results['mae_mean']:.3f} (+/- {cv_results['mae_std']:.3f})")
    print(f"  R²:   {cv_results['r2_mean']:.3f} (+/- {cv_results['r2_std']:.3f})")
```

### Example 4: Custom Feature Engineering

```python
from nba_api.stats_modeling.utils import FeatureEngineer
from nba_api.stats_modeling import PointsPredictor
import pandas as pd

# Custom feature engineer with different windows
fe = FeatureEngineer(
    rolling_windows=[5, 10, 15, 30],  # Custom windows
    include_advanced=True
)

# Load data
from nba_api.stats.endpoints import PlayerGameLog
logs = PlayerGameLog(player_id=2544, season="2024-25")
df = logs.get_data_frames()[0]

# Engineer features
engineered = fe.engineer_player_features(df, target_col='PTS')

# Train model with custom features
model = PointsPredictor()
model.feature_engineer = fe  # Use custom engineer
metrics = model.fit(engineered, target_col='PTS', engineer_features=False)

print(f"Custom model RMSE: {metrics['validation_metrics']['rmse']:.2f}")
```

---

## Best Practices

### 1. Data Collection

```python
# ✓ Do: Use recent seasons for relevant predictions
predictor.collect_data(seasons=['2024-25', '2023-24'])

# ✗ Don't: Use too many old seasons (player styles change)
predictor.collect_data(seasons=['2024-25', '2023-24', '2022-23',
                                '2021-22', '2020-21', '2019-20'])
```

### 2. Training

```python
# ✓ Do: Use time-series cross-validation
cv_results = model.cross_validate(df, target_col='PTS', n_splits=5)

# ✗ Don't: Use random train/test splits (causes data leakage)
# The models automatically use time-ordered splits
```

### 3. Predictions

```python
# ✓ Do: Check prediction confidence
pred, lower, upper = model.predict_with_confidence(df)
if (upper - lower) > 20:  # High uncertainty
    print("Warning: High prediction uncertainty")

# ✓ Do: Use recent context
result = predictor.predict_points(player_id, n_games_context=15)

# ✗ Don't: Predict without enough recent data
# Models need at least 5 games of context
```

### 4. Model Updates

```python
# ✓ Do: Retrain models periodically
# Weekly retraining captures recent form
predictor.collect_data(seasons=['2024-25'])
predictor.train_all_models()
predictor.save_all_models()

# ✓ Do: Monitor model performance
metrics = model.cross_validate(df, target_col='PTS')
if metrics['rmse_mean'] > 8.0:
    print("Warning: Model performance degraded, consider retraining")
```

### 5. Production Usage

```python
# ✓ Do: Load models once, predict many times
predictor = NBAPredictor()
predictor.load_all_models()
predictor.load_data()

# Reuse for multiple predictions
for player_id in player_ids:
    result = predictor.predict_points(player_id)

# ✗ Don't: Reload models for each prediction
```

---

## Troubleshooting

### Common Issues

**"No module named 'sklearn'"**
```bash
pip install nba_api[modeling]
```

**"Insufficient training data"**
- Ensure you've collected enough seasons
- Some players may not have enough games

**"Model must be fitted first"**
```python
# Train or load models before predicting
predictor.train_all_models()
# or
predictor.load_all_models()
```

**API Rate Limiting**
- The collector automatically handles rate limiting
- Increase delay if still getting errors:
```python
collector = NBADataCollector(request_delay=1.0)
```

---

## License

MIT License - see the main nba_api package license.

## Contributing

Contributions are welcome! Please see the main nba_api repository for contribution guidelines.
