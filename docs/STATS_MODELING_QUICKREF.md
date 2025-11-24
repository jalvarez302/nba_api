# NBA Stats Modeling - Quick Reference

## Installation

```bash
pip install nba_api[modeling]           # Basic
pip install nba_api[modeling-advanced]  # + LightGBM, XGBoost
```

## Quick Start

```python
from nba_api.stats_modeling import NBAPredictor

predictor = NBAPredictor()
predictor.collect_data(seasons=['2024-25', '2023-24'])
predictor.train_all_models()

# Predictions
pts = predictor.predict_points(player_id=2544)
game = predictor.predict_game(home_team_id=1610612747, away_team_id=1610612744)
stats = predictor.predict_all_stats(player_id=2544)
```

## Player ID Lookup

```python
player_id = predictor.get_player_id("LeBron James")
team_id = predictor.get_team_id("Lakers")
```

## Common Player IDs

| Player | ID |
|--------|-----|
| LeBron James | 2544 |
| Stephen Curry | 201939 |
| Kevin Durant | 201142 |
| Giannis Antetokounmpo | 203507 |
| Luka Doncic | 1629029 |
| Nikola Jokic | 203999 |
| Joel Embiid | 203954 |
| Jayson Tatum | 1628369 |

## Common Team IDs

| Team | ID |
|------|-----|
| Lakers | 1610612747 |
| Warriors | 1610612744 |
| Celtics | 1610612738 |
| Nuggets | 1610612743 |
| Heat | 1610612748 |
| Bucks | 1610612749 |
| 76ers | 1610612755 |
| Suns | 1610612756 |

## Prediction Methods

| Method | Returns |
|--------|---------|
| `predict_points(player_id)` | `{predicted_points, confidence_lower, confidence_upper, recent_average}` |
| `predict_game(home_id, away_id)` | `{home_win_probability, away_win_probability, predicted_winner, confidence}` |
| `predict_all_stats(player_id)` | `{predictions: {FG3M, FTM, BLK, FGM}, recent_averages: {...}}` |
| `predict_three_pointers(player_id)` | `{predictions: {FG3M}, recent_averages: {...}}` |
| `predict_free_throws(player_id)` | `{predictions: {FTM}, recent_averages: {...}}` |
| `predict_blocks(player_id)` | `{predictions: {BLK}, recent_averages: {...}}` |
| `predict_field_goals(player_id)` | `{predictions: {FGM}, recent_averages: {...}}` |

## Save & Load Models

```python
predictor.save_all_models()  # Save to disk
predictor.load_all_models()  # Load from disk
```

## Direct Model Usage

```python
from nba_api.stats_modeling import PointsPredictor, GameOutcomePredictor

# Points model
pts_model = PointsPredictor(backend='sklearn')
pts_model.fit(game_logs_df, target_col='PTS')
predictions = pts_model.predict(new_df)

# Game outcome model
game_model = GameOutcomePredictor()
game_model.fit(team_logs_df, target_col='WIN')
win_prob = game_model.predict_proba(team_df)
```

## Feature Engineering

```python
from nba_api.stats_modeling.utils import FeatureEngineer

fe = FeatureEngineer()
df = fe.engineer_player_features(raw_df, target_col='PTS')
X, y = fe.prepare_training_data(df, target_col='PTS')
```

## Cross-Validation

```python
cv_results = model.cross_validate(df, target_col='PTS', n_splits=5)
print(f"RMSE: {cv_results['rmse_mean']:.2f} (+/- {cv_results['rmse_std']:.2f})")
```

## Data Storage

```python
from nba_api.stats_modeling import NBADataStorage

storage = NBADataStorage("./nba_data")
df = storage.load_dataset("league_player_game_logs")
players = storage.get_players(active_only=True)
games = storage.get_games(season='2024-25', team_id=1610612747)
stats = storage.get_storage_stats()
```

## Model Backends

| Backend | Speed | Accuracy | Install |
|---------|-------|----------|---------|
| sklearn | Medium | Good | Built-in |
| lightgbm | Fast | Better | `pip install lightgbm` |
| xgboost | Medium | Better | `pip install xgboost` |

```python
model = PointsPredictor(backend='lightgbm')  # Use LightGBM
```
