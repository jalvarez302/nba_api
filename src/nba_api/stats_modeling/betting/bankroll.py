"""
Bankroll Management and Bet Sizing

Professional money management for sports betting:
- Kelly Criterion for optimal bet sizing
- Fractional Kelly for risk management
- Bankroll tracking and analysis
- Risk of ruin calculations
- Unit-based betting systems
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import pandas as pd

from nba_api.stats_modeling.betting.odds import (
    american_to_decimal,
    implied_probability,
    calculate_ev,
)


def kelly_criterion(
    win_probability: float,
    odds: float,
    odds_format: str = 'decimal'
) -> float:
    """
    Calculate Kelly Criterion bet size.

    The Kelly Criterion maximizes long-term growth rate but can be
    aggressive. Use fractional Kelly for more conservative sizing.

    Formula: f* = (bp - q) / b
    Where:
        f* = fraction of bankroll to bet
        b = net odds (decimal - 1)
        p = probability of winning
        q = probability of losing (1 - p)

    Args:
        win_probability: Your estimated probability of winning (0-1)
        odds: The odds offered
        odds_format: 'decimal' or 'american'

    Returns:
        Optimal fraction of bankroll to bet (0-1)
        Returns 0 if bet has negative expected value

    Examples:
        >>> kelly_criterion(0.55, 2.0)  # 55% at even money
        0.10  # Bet 10% of bankroll
        >>> kelly_criterion(0.52, -110, 'american')
        0.016  # Bet 1.6% of bankroll
    """
    if odds_format == 'american':
        decimal_odds = american_to_decimal(odds)
    else:
        decimal_odds = odds

    b = decimal_odds - 1  # Net odds
    p = win_probability
    q = 1 - p

    kelly = (b * p - q) / b

    # Never recommend negative bets
    return max(0, kelly)


def fractional_kelly(
    win_probability: float,
    odds: float,
    fraction: float = 0.25,
    odds_format: str = 'decimal'
) -> float:
    """
    Calculate fractional Kelly bet size.

    Fractional Kelly reduces variance at the cost of some expected growth.
    Common fractions:
    - 1.0 (full Kelly): Maximum growth, high variance
    - 0.5 (half Kelly): Popular choice, balanced
    - 0.25 (quarter Kelly): Conservative, lower drawdowns
    - 0.1 (tenth Kelly): Very conservative

    Args:
        win_probability: Estimated win probability
        odds: Odds offered
        fraction: Kelly fraction to use (0-1)
        odds_format: Odds format

    Returns:
        Bet size as fraction of bankroll
    """
    full_kelly = kelly_criterion(win_probability, odds, odds_format)
    return full_kelly * fraction


def calculate_bet_size(
    bankroll: float,
    win_probability: float,
    odds: float,
    method: str = 'fractional_kelly',
    kelly_fraction: float = 0.25,
    max_bet_pct: float = 0.05,
    min_bet_pct: float = 0.01,
    odds_format: str = 'decimal'
) -> Dict:
    """
    Calculate recommended bet size with multiple methods.

    Args:
        bankroll: Current bankroll amount
        win_probability: Estimated win probability
        odds: Odds offered
        method: 'kelly', 'fractional_kelly', 'fixed', or 'confidence'
        kelly_fraction: Fraction of Kelly to use
        max_bet_pct: Maximum bet as percentage of bankroll
        min_bet_pct: Minimum bet as percentage of bankroll
        odds_format: Odds format

    Returns:
        Dictionary with bet sizing recommendation
    """
    if odds_format == 'american':
        decimal_odds = american_to_decimal(odds)
    else:
        decimal_odds = odds

    # Calculate EV first
    ev = calculate_ev(win_probability, decimal_odds, 'decimal')

    if ev <= 0:
        return {
            'recommended_bet': 0,
            'bet_percentage': 0,
            'expected_value': ev,
            'reason': 'Negative EV - no bet recommended',
            'method': method,
        }

    # Calculate Kelly
    full_kelly = kelly_criterion(win_probability, decimal_odds, 'decimal')

    if method == 'kelly':
        bet_pct = full_kelly
    elif method == 'fractional_kelly':
        bet_pct = full_kelly * kelly_fraction
    elif method == 'fixed':
        bet_pct = min_bet_pct
    elif method == 'confidence':
        # Scale bet size by confidence (EV magnitude)
        confidence = min(1.0, ev * 10)  # Scale EV to 0-1
        bet_pct = min_bet_pct + (max_bet_pct - min_bet_pct) * confidence
    else:
        bet_pct = full_kelly * kelly_fraction

    # Apply constraints
    bet_pct = max(min_bet_pct, min(max_bet_pct, bet_pct))
    bet_amount = bankroll * bet_pct

    return {
        'recommended_bet': round(bet_amount, 2),
        'bet_percentage': bet_pct,
        'full_kelly': full_kelly,
        'expected_value': ev,
        'ev_percent': ev * 100 / 1,  # EV as % of stake
        'potential_profit': round(bet_amount * (decimal_odds - 1), 2),
        'method': method,
        'edge': win_probability - implied_probability(decimal_odds),
    }


def risk_of_ruin(
    win_probability: float,
    odds: float,
    bet_fraction: float,
    target_multiple: float = 2.0,
    odds_format: str = 'decimal'
) -> float:
    """
    Calculate probability of losing entire bankroll before reaching target.

    Uses simplified formula for even-money approximation.
    More accurate for realistic bet sizing (small fractions).

    Args:
        win_probability: Win probability
        odds: Odds
        bet_fraction: Fraction of bankroll per bet
        target_multiple: Target bankroll multiple (e.g., 2.0 = double bankroll)
        odds_format: Odds format

    Returns:
        Probability of ruin (0-1)
    """
    if odds_format == 'american':
        decimal_odds = american_to_decimal(odds)
    else:
        decimal_odds = odds

    # Edge
    p = win_probability
    q = 1 - p
    b = decimal_odds - 1

    # Expected growth per bet
    edge = p * b - q

    if edge <= 0:
        return 1.0  # Guaranteed ruin with negative edge

    # Simplified risk of ruin for small bet sizes
    # R ≈ ((1-edge)/(1+edge))^n where n = 1/bet_fraction
    n = 1 / bet_fraction

    if edge >= 1:
        return 0.0

    ror = ((1 - edge) / (1 + edge)) ** n

    return min(1.0, ror)


@dataclass
class Bet:
    """Represents a single bet."""
    timestamp: datetime
    event: str
    selection: str
    odds: float
    odds_format: str
    stake: float
    win_probability: float
    result: Optional[str] = None  # 'win', 'loss', 'push', 'pending'
    profit: float = 0.0
    notes: str = ""

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'event': self.event,
            'selection': self.selection,
            'odds': self.odds,
            'odds_format': self.odds_format,
            'stake': self.stake,
            'win_probability': self.win_probability,
            'result': self.result,
            'profit': self.profit,
            'notes': self.notes,
        }


@dataclass
class BankrollState:
    """Snapshot of bankroll at a point in time."""
    timestamp: datetime
    balance: float
    total_wagered: float
    total_profit: float
    num_bets: int
    win_rate: float
    roi: float


class BankrollManager:
    """
    Comprehensive bankroll management system.

    Features:
    - Track all bets and results
    - Calculate statistics and ROI
    - Recommend bet sizes
    - Monitor for bankroll warnings
    """

    def __init__(
        self,
        initial_bankroll: float,
        kelly_fraction: float = 0.25,
        max_bet_pct: float = 0.05,
        min_bet_pct: float = 0.01,
        unit_size: Optional[float] = None
    ):
        """
        Initialize bankroll manager.

        Args:
            initial_bankroll: Starting bankroll
            kelly_fraction: Fraction of Kelly to use
            max_bet_pct: Maximum single bet as % of bankroll
            min_bet_pct: Minimum bet as % of bankroll
            unit_size: Fixed unit size (overrides percentage-based)
        """
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.kelly_fraction = kelly_fraction
        self.max_bet_pct = max_bet_pct
        self.min_bet_pct = min_bet_pct
        self.unit_size = unit_size

        self.bets: List[Bet] = []
        self.bankroll_history: List[BankrollState] = []

        # Record initial state
        self._record_state()

    def _record_state(self):
        """Record current bankroll state."""
        completed_bets = [b for b in self.bets if b.result in ('win', 'loss')]

        if completed_bets:
            wins = sum(1 for b in completed_bets if b.result == 'win')
            win_rate = wins / len(completed_bets)
        else:
            win_rate = 0.0

        total_wagered = sum(b.stake for b in completed_bets)
        total_profit = self.current_bankroll - self.initial_bankroll

        roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0.0

        self.bankroll_history.append(BankrollState(
            timestamp=datetime.now(),
            balance=self.current_bankroll,
            total_wagered=total_wagered,
            total_profit=total_profit,
            num_bets=len(completed_bets),
            win_rate=win_rate,
            roi=roi,
        ))

    def recommend_bet(
        self,
        win_probability: float,
        odds: float,
        odds_format: str = 'decimal'
    ) -> Dict:
        """
        Get bet sizing recommendation.

        Args:
            win_probability: Your estimated win probability
            odds: Odds offered
            odds_format: Odds format

        Returns:
            Dictionary with recommendation
        """
        return calculate_bet_size(
            bankroll=self.current_bankroll,
            win_probability=win_probability,
            odds=odds,
            method='fractional_kelly',
            kelly_fraction=self.kelly_fraction,
            max_bet_pct=self.max_bet_pct,
            min_bet_pct=self.min_bet_pct,
            odds_format=odds_format
        )

    def place_bet(
        self,
        event: str,
        selection: str,
        odds: float,
        stake: float,
        win_probability: float,
        odds_format: str = 'decimal',
        notes: str = ""
    ) -> Bet:
        """
        Record a bet placement.

        Args:
            event: Event description
            selection: Your selection
            odds: Odds
            stake: Amount wagered
            win_probability: Your estimated probability
            odds_format: Odds format
            notes: Optional notes

        Returns:
            Bet object
        """
        bet = Bet(
            timestamp=datetime.now(),
            event=event,
            selection=selection,
            odds=odds,
            odds_format=odds_format,
            stake=stake,
            win_probability=win_probability,
            result='pending',
            notes=notes,
        )

        self.bets.append(bet)
        return bet

    def record_result(
        self,
        bet_index: int,
        result: str,  # 'win', 'loss', 'push'
    ):
        """
        Record the result of a bet.

        Args:
            bet_index: Index of bet in self.bets
            result: 'win', 'loss', or 'push'
        """
        bet = self.bets[bet_index]
        bet.result = result

        if bet.odds_format == 'american':
            decimal_odds = american_to_decimal(bet.odds)
        else:
            decimal_odds = bet.odds

        if result == 'win':
            bet.profit = bet.stake * (decimal_odds - 1)
            self.current_bankroll += bet.profit
        elif result == 'loss':
            bet.profit = -bet.stake
            self.current_bankroll -= bet.stake
        else:  # push
            bet.profit = 0

        self._record_state()

    def get_statistics(self) -> Dict:
        """Get comprehensive betting statistics."""
        completed = [b for b in self.bets if b.result in ('win', 'loss', 'push')]

        if not completed:
            return {
                'total_bets': 0,
                'pending_bets': len(self.bets),
                'current_bankroll': self.current_bankroll,
                'total_profit': 0,
                'roi': 0,
            }

        wins = [b for b in completed if b.result == 'win']
        losses = [b for b in completed if b.result == 'loss']

        total_wagered = sum(b.stake for b in completed)
        total_profit = sum(b.profit for b in completed)

        # Calculate CLV (Closing Line Value) if we tracked closing odds
        avg_edge = np.mean([
            b.win_probability - implied_probability(
                b.odds if b.odds_format == 'decimal' else american_to_decimal(b.odds)
            )
            for b in completed
        ])

        return {
            'total_bets': len(completed),
            'wins': len(wins),
            'losses': len(losses),
            'pushes': len(completed) - len(wins) - len(losses),
            'win_rate': len(wins) / len(completed) if completed else 0,
            'total_wagered': total_wagered,
            'total_profit': total_profit,
            'roi': (total_profit / total_wagered * 100) if total_wagered > 0 else 0,
            'current_bankroll': self.current_bankroll,
            'bankroll_growth': (self.current_bankroll / self.initial_bankroll - 1) * 100,
            'average_stake': total_wagered / len(completed),
            'average_odds': np.mean([
                b.odds if b.odds_format == 'decimal' else american_to_decimal(b.odds)
                for b in completed
            ]),
            'average_edge': avg_edge,
            'max_drawdown': self._calculate_max_drawdown(),
            'sharpe_ratio': self._calculate_sharpe(),
        }

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from peak."""
        if len(self.bankroll_history) < 2:
            return 0.0

        balances = [s.balance for s in self.bankroll_history]
        peak = balances[0]
        max_dd = 0.0

        for balance in balances:
            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak
            max_dd = max(max_dd, dd)

        return max_dd * 100

    def _calculate_sharpe(self) -> float:
        """Calculate Sharpe-like ratio for betting."""
        completed = [b for b in self.bets if b.result in ('win', 'loss')]

        if len(completed) < 2:
            return 0.0

        returns = [b.profit / b.stake for b in completed]
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        return mean_return / std_return * np.sqrt(len(completed))

    def get_bet_history(self) -> pd.DataFrame:
        """Get bet history as DataFrame."""
        return pd.DataFrame([b.to_dict() for b in self.bets])

    def get_bankroll_history(self) -> pd.DataFrame:
        """Get bankroll history as DataFrame."""
        return pd.DataFrame([
            {
                'timestamp': s.timestamp,
                'balance': s.balance,
                'total_wagered': s.total_wagered,
                'total_profit': s.total_profit,
                'num_bets': s.num_bets,
                'win_rate': s.win_rate,
                'roi': s.roi,
            }
            for s in self.bankroll_history
        ])

    def check_warnings(self) -> List[str]:
        """Check for bankroll warning conditions."""
        warnings = []

        # Drawdown warning
        max_dd = self._calculate_max_drawdown()
        if max_dd > 20:
            warnings.append(f"High drawdown: {max_dd:.1f}%")

        # Bankroll depletion warning
        depletion = (1 - self.current_bankroll / self.initial_bankroll) * 100
        if depletion > 30:
            warnings.append(f"Significant bankroll depletion: {depletion:.1f}%")

        # Losing streak warning
        recent = self.bets[-10:] if len(self.bets) >= 10 else self.bets
        recent_completed = [b for b in recent if b.result in ('win', 'loss')]
        if len(recent_completed) >= 5:
            recent_losses = sum(1 for b in recent_completed if b.result == 'loss')
            if recent_losses / len(recent_completed) > 0.7:
                warnings.append("Recent losing streak detected")

        return warnings
