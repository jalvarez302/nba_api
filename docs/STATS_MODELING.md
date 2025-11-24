# NBA Statistics Modeling Package v2.0

**Professional Sports Betting Analytics Platform**

A comprehensive ML system for NBA predictions built with methodologies used by professional sports betting operations.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Prediction Models](#prediction-models)
4. [Betting Utilities](#betting-utilities)
5. [Backtesting](#backtesting)
6. [API Reference](#api-reference)

---

## Quick Start

```python
from nba_api.stats_modeling import (
    NBAPredictor,
    calculate_ev,
    BankrollManager,
    kelly_criterion,
)

# 1. Setup and collect data
predictor = NBAPredictor()
predictor.collect_data(seasons=['2024-25', '2023-24'])
predictor.train_all_models()

# 2. Make predictions
lebron_id = predictor.get_player_id("LeBron James")
points = predictor.predict_points(lebron_id)
print(f"Predicted: {points['predicted_points']:.1f} pts")
print(f"Range: [{points['confidence_lower']:.1f}, {points['confidence_upper']:.1f}]")

# 3. Calculate betting value
my_probability = 0.58  # My estimated over probability
book_odds = -110       # American odds

ev = calculate_ev(my_probability, book_odds, 'american')
print(f"Expected Value: {ev*100:.2f}%")

# 4. Determine bet size
bet_pct = kelly_criterion(my_probability, book_odds, 'american')
print(f"Kelly suggests: {bet_pct*100:.1f}% of bankroll")

# 5. Save models for later
predictor.save_all_models()
```

---

## Installation

### Basic (sklearn only)
```bash
pip install nba_api[modeling]
```

### Advanced (+ LightGBM & XGBoost)
```bash
pip install nba_api[modeling-advanced]
```

---

## Prediction Models

### Player Points
```python
from nba_api.stats_modeling import PointsPredictor

model = PointsPredictor(backend='sklearn')
model.fit(game_logs_df, target_col='PTS')
pred, lower, upper = model.predict_with_confidence(new_df)
```

### Player Props (Over/Under)
```python
from nba_api.stats_modeling import PlayerPropsPredictor

props = PlayerPropsPredictor(prop_type='PTS')  # or 'REB', 'AST', 'PTS_REB_AST'
props.fit(game_logs_df)

result = props.predict_player_prop(
    player_id=2544,
    game_logs_df=df,
    line=24.5
)
print(f"Over probability: {result['over_probability']:.1%}")
```

### Point Spreads
```python
from nba_api.stats_modeling import SpreadPredictor

spread = SpreadPredictor(use_ensemble=True)
spread.fit(team_logs_df)

result = spread.predict_spread_probability(
    home_team_logs, away_team_logs, line=-3.5
)
print(f"Home cover prob: {result['home_cover_probability']:.1%}")
```

### Game Totals
```python
from nba_api.stats_modeling import TotalsPredictor

totals = TotalsPredictor()
totals.fit(team_logs_df)

result = totals.predict_total_probability(
    home_logs, away_logs, line=220.5
)
print(f"Over prob: {result['over_probability']:.1%}")
```

### Ensemble Models
```python
from nba_api.stats_modeling import (
    StackedEnsembleRegressor,
    BlendingEnsemble,
    BayesianModelAveraging,
)

# Stacking (highest accuracy)
stacked = StackedEnsembleRegressor(use_neural_net=True)
stacked.fit(X_train, y_train)

# Bayesian (with uncertainty)
bayesian = BayesianModelAveraging(n_bootstrap=100)
bayesian.fit(X, y)
mean, lower, upper = bayesian.predict_with_uncertainty(X_test)
```

---

## Betting Utilities

### Odds Conversion
```python
from nba_api.stats_modeling import (
    american_to_decimal,
    decimal_to_american,
    implied_probability,
    OddsConverter,
)

# Quick conversion
decimal = american_to_decimal(-110)  # 1.909
american = decimal_to_american(2.0)   # +100
prob = implied_probability(-110, 'american')  # 52.4%

# Full converter
odds = OddsConverter.from_american(-110)
print(f"Decimal: {odds.to_decimal():.3f}")
print(f"Implied: {odds.to_probability():.1%}")
```

### Expected Value
```python
from nba_api.stats_modeling import calculate_ev

# Positive EV = profitable bet
ev = calculate_ev(
    win_probability=0.55,
    odds=-110,
    odds_format='american'
)
print(f"EV: {ev*100:.2f}%")
```

### Kelly Criterion
```python
from nba_api.stats_modeling import kelly_criterion, fractional_kelly

# Full Kelly (aggressive)
full = kelly_criterion(0.55, 2.0)  # 10%

# Fractional Kelly (recommended)
quarter = fractional_kelly(0.55, 2.0, fraction=0.25)  # 2.5%
```

### Bankroll Management
```python
from nba_api.stats_modeling import BankrollManager

bankroll = BankrollManager(
    initial_bankroll=10000,
    kelly_fraction=0.25,
    max_bet_pct=0.05
)

# Get recommendation
rec = bankroll.recommend_bet(0.55, -110, 'american')
print(f"Bet: ${rec['recommended_bet']:.2f}")
print(f"EV: {rec['expected_value']*100:.2f}%")

# Track bets
bankroll.place_bet("Lakers -3.5", "spread", -110, 250, 0.55, 'american')
bankroll.record_result(0, 'win')

# Statistics
stats = bankroll.get_statistics()
print(f"ROI: {stats['roi']:.1f}%")
```

---

## Backtesting

```python
from nba_api.stats_modeling import Backtester, PlayerPropsPredictor

backtester = Backtester(
    initial_bankroll=10000,
    bet_sizing='fractional_kelly',
    kelly_fraction=0.25
)

model = PlayerPropsPredictor(prop_type='PTS')
results = backtester.backtest_props(
    model=model,
    data=game_logs_df,
    lines=lines_dict,
    actuals=actuals_dict,
    min_edge=0.03
)

print(results.summary())
# Total Bets: 500
# Win Rate: 54.2%
# ROI: 8.3%
# Max Drawdown: 12.1%
# Sharpe: 1.45

# Monte Carlo simulation
mc = backtester.monte_carlo_analysis(results, n_simulations=1000)
print(f"P(profit): {mc['probability_of_profit']:.1%}")
```

---

## API Reference

### Models

| Class | Purpose |
|-------|---------|
| `NBAPredictor` | High-level unified interface |
| `PointsPredictor` | Player points prediction |
| `PlayerPropsPredictor` | Player prop betting |
| `SpreadPredictor` | Point spread prediction |
| `TotalsPredictor` | Over/under totals |
| `GameOutcomePredictor` | Win probability |
| `StackedEnsembleRegressor` | Stacking ensemble |
| `BlendingEnsemble` | Optimized blending |
| `BayesianModelAveraging` | Bayesian uncertainty |

### Betting

| Class/Function | Purpose |
|----------------|---------|
| `OddsConverter` | Odds conversion |
| `calculate_ev` | Expected value |
| `kelly_criterion` | Full Kelly sizing |
| `fractional_kelly` | Conservative Kelly |
| `BankrollManager` | Complete bankroll system |
| `Backtester` | Strategy backtesting |

### Common IDs

| Player | ID |
|--------|-----|
| LeBron James | 2544 |
| Stephen Curry | 201939 |
| Giannis Antetokounmpo | 203507 |
| Luka Doncic | 1629029 |

| Team | ID |
|------|-----|
| Lakers | 1610612747 |
| Warriors | 1610612744 |
| Celtics | 1610612738 |

---

## Best Practices

1. **Use fractional Kelly** (25%) - Full Kelly is too aggressive
2. **Minimum 3% edge** before betting
3. **Retrain weekly** to capture recent form
4. **Time-series CV** - Never use random splits
5. **Backtest first** before live betting
6. **Track all bets** for performance analysis
