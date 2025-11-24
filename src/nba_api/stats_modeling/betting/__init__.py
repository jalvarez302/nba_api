"""
Betting utilities for NBA statistics modeling.

Provides:
- Odds conversion between formats
- Expected value calculation
- Kelly criterion for bet sizing
- Bankroll management
- Backtesting framework
"""

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

__all__ = [
    'OddsConverter',
    'american_to_decimal',
    'decimal_to_american',
    'implied_probability',
    'calculate_ev',
    'calculate_vig',
    'no_vig_odds',
    'BankrollManager',
    'kelly_criterion',
    'fractional_kelly',
    'Backtester',
    'BacktestResult',
]
