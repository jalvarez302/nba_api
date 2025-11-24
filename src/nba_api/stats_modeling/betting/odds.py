"""
Odds Conversion and Expected Value Utilities

Professional-grade tools for working with betting odds:
- Convert between American, Decimal, and Fractional formats
- Calculate implied probabilities
- Remove bookmaker vig
- Calculate expected value
- Compare odds across books
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def american_to_decimal(american_odds: float) -> float:
    """
    Convert American odds to decimal odds.

    American odds:
    - Positive (+150): Amount won on $100 bet
    - Negative (-150): Amount needed to bet to win $100

    Examples:
        >>> american_to_decimal(150)   # +150
        2.5
        >>> american_to_decimal(-150)  # -150
        1.667
    """
    if american_odds >= 0:
        return 1 + american_odds / 100
    else:
        return 1 + 100 / abs(american_odds)


def decimal_to_american(decimal_odds: float) -> float:
    """
    Convert decimal odds to American odds.

    Examples:
        >>> decimal_to_american(2.5)
        150.0
        >>> decimal_to_american(1.5)
        -200.0
    """
    if decimal_odds >= 2.0:
        return (decimal_odds - 1) * 100
    else:
        return -100 / (decimal_odds - 1)


def fractional_to_decimal(numerator: float, denominator: float) -> float:
    """
    Convert fractional odds to decimal.

    Examples:
        >>> fractional_to_decimal(3, 2)  # 3/2
        2.5
        >>> fractional_to_decimal(1, 2)  # 1/2
        1.5
    """
    return 1 + numerator / denominator


def decimal_to_fractional(decimal_odds: float) -> Tuple[int, int]:
    """
    Convert decimal odds to fractional (approximate).

    Returns numerator and denominator.
    """
    from fractions import Fraction
    frac = Fraction(decimal_odds - 1).limit_denominator(100)
    return frac.numerator, frac.denominator


def implied_probability(odds: float, odds_format: str = 'decimal') -> float:
    """
    Calculate implied probability from odds.

    Args:
        odds: The odds value
        odds_format: 'decimal', 'american', or 'probability'

    Returns:
        Implied probability (0-1)

    Examples:
        >>> implied_probability(2.0)  # Even odds
        0.5
        >>> implied_probability(-110, 'american')
        0.524
    """
    if odds_format == 'american':
        decimal_odds = american_to_decimal(odds)
    elif odds_format == 'probability':
        return odds
    else:
        decimal_odds = odds

    return 1 / decimal_odds


def probability_to_odds(probability: float, odds_format: str = 'decimal') -> float:
    """
    Convert probability to odds.

    Args:
        probability: Win probability (0-1)
        odds_format: Output format ('decimal' or 'american')

    Returns:
        Odds in requested format
    """
    if probability <= 0 or probability >= 1:
        raise ValueError("Probability must be between 0 and 1")

    decimal_odds = 1 / probability

    if odds_format == 'american':
        return decimal_to_american(decimal_odds)
    return decimal_odds


def calculate_vig(
    odds_1: float,
    odds_2: float,
    odds_format: str = 'decimal'
) -> float:
    """
    Calculate the bookmaker's vig (vigorish/juice).

    For a two-way market, vig = total implied probability - 1

    Args:
        odds_1: Odds for outcome 1
        odds_2: Odds for outcome 2
        odds_format: Format of input odds

    Returns:
        Vig as a decimal (e.g., 0.05 for 5% vig)

    Examples:
        >>> calculate_vig(-110, -110, 'american')
        0.0476  # ~4.76% vig (standard juice)
    """
    prob_1 = implied_probability(odds_1, odds_format)
    prob_2 = implied_probability(odds_2, odds_format)

    return prob_1 + prob_2 - 1


def no_vig_odds(
    odds_1: float,
    odds_2: float,
    odds_format: str = 'decimal'
) -> Tuple[float, float]:
    """
    Remove vig to get fair odds.

    Args:
        odds_1: Odds for outcome 1
        odds_2: Odds for outcome 2
        odds_format: Format of input odds

    Returns:
        Tuple of (fair_odds_1, fair_odds_2) in decimal format
    """
    prob_1 = implied_probability(odds_1, odds_format)
    prob_2 = implied_probability(odds_2, odds_format)

    total = prob_1 + prob_2

    fair_prob_1 = prob_1 / total
    fair_prob_2 = prob_2 / total

    return (1 / fair_prob_1, 1 / fair_prob_2)


def calculate_ev(
    win_probability: float,
    odds: float,
    odds_format: str = 'decimal',
    stake: float = 1.0
) -> float:
    """
    Calculate expected value of a bet.

    EV = (prob_win * profit) - (prob_lose * stake)

    Args:
        win_probability: Your estimated probability of winning (0-1)
        odds: The odds offered
        odds_format: Format of odds
        stake: Bet amount (default 1 unit)

    Returns:
        Expected value (positive = +EV bet)

    Examples:
        >>> calculate_ev(0.55, 2.0)  # 55% chance at even money
        0.10  # +10% EV
        >>> calculate_ev(0.52, -110, 'american')
        0.0073  # +0.73% EV
    """
    if odds_format == 'american':
        decimal_odds = american_to_decimal(odds)
    else:
        decimal_odds = odds

    profit_if_win = stake * (decimal_odds - 1)
    loss_if_lose = stake

    ev = (win_probability * profit_if_win) - ((1 - win_probability) * loss_if_lose)

    return ev


def calculate_ev_percent(
    win_probability: float,
    odds: float,
    odds_format: str = 'decimal'
) -> float:
    """
    Calculate expected value as a percentage of stake.

    Returns:
        EV as percentage (e.g., 5.0 for 5% EV)
    """
    return calculate_ev(win_probability, odds, odds_format, stake=100)


def break_even_probability(odds: float, odds_format: str = 'decimal') -> float:
    """
    Calculate the break-even win probability for given odds.

    This is the minimum win rate needed to profit long-term.

    Examples:
        >>> break_even_probability(-110, 'american')
        0.524  # Need 52.4% to break even at -110
        >>> break_even_probability(2.0)
        0.5  # Need 50% at even money
    """
    return implied_probability(odds, odds_format)


class OddsConverter:
    """
    Comprehensive odds conversion and analysis tool.

    Supports American, Decimal, and Fractional odds.
    """

    def __init__(self, odds: float, odds_format: str = 'decimal'):
        """
        Initialize with odds in any format.

        Args:
            odds: The odds value
            odds_format: 'american', 'decimal', or 'fractional_decimal'
        """
        if odds_format == 'american':
            self.decimal = american_to_decimal(odds)
            self.american = odds
        elif odds_format == 'decimal':
            self.decimal = odds
            self.american = decimal_to_american(odds)
        else:
            self.decimal = odds
            self.american = decimal_to_american(odds)

        self.implied_prob = 1 / self.decimal
        self.fractional = decimal_to_fractional(self.decimal)

    @classmethod
    def from_american(cls, odds: float) -> 'OddsConverter':
        return cls(odds, 'american')

    @classmethod
    def from_decimal(cls, odds: float) -> 'OddsConverter':
        return cls(odds, 'decimal')

    @classmethod
    def from_probability(cls, prob: float) -> 'OddsConverter':
        if prob <= 0 or prob >= 1:
            raise ValueError("Probability must be between 0 and 1")
        return cls(1 / prob, 'decimal')

    def to_american(self) -> float:
        return self.american

    def to_decimal(self) -> float:
        return self.decimal

    def to_fractional(self) -> Tuple[int, int]:
        return self.fractional

    def to_probability(self) -> float:
        return self.implied_prob

    def profit_on_stake(self, stake: float) -> float:
        """Calculate profit for a given stake if bet wins."""
        return stake * (self.decimal - 1)

    def payout_on_stake(self, stake: float) -> float:
        """Calculate total payout (stake + profit) if bet wins."""
        return stake * self.decimal

    def calculate_ev(self, true_probability: float, stake: float = 1.0) -> float:
        """Calculate expected value given true win probability."""
        return calculate_ev(true_probability, self.decimal, 'decimal', stake)

    def edge(self, true_probability: float) -> float:
        """
        Calculate edge over the book.

        Edge = true_probability - implied_probability
        """
        return true_probability - self.implied_prob

    def __repr__(self) -> str:
        sign = '+' if self.american > 0 else ''
        return (f"OddsConverter(decimal={self.decimal:.3f}, "
                f"american={sign}{self.american:.0f}, "
                f"implied_prob={self.implied_prob:.1%})")


class OddsComparison:
    """
    Compare odds across multiple sportsbooks.

    Useful for line shopping and finding the best value.
    """

    def __init__(self):
        self.odds_by_book: Dict[str, Dict[str, OddsConverter]] = {}

    def add_odds(
        self,
        book_name: str,
        outcome: str,
        odds: float,
        odds_format: str = 'american'
    ):
        """Add odds from a sportsbook."""
        if book_name not in self.odds_by_book:
            self.odds_by_book[book_name] = {}

        self.odds_by_book[book_name][outcome] = OddsConverter(odds, odds_format)

    def best_odds(self, outcome: str) -> Tuple[str, OddsConverter]:
        """Find the best odds for an outcome."""
        best_book = None
        best_odds = None

        for book, outcomes in self.odds_by_book.items():
            if outcome in outcomes:
                if best_odds is None or outcomes[outcome].decimal > best_odds.decimal:
                    best_book = book
                    best_odds = outcomes[outcome]

        return best_book, best_odds

    def compare(self, outcome: str) -> pd.DataFrame:
        """Compare odds across all books for an outcome."""
        import pandas as pd

        rows = []
        for book, outcomes in self.odds_by_book.items():
            if outcome in outcomes:
                conv = outcomes[outcome]
                rows.append({
                    'book': book,
                    'american': conv.american,
                    'decimal': conv.decimal,
                    'implied_prob': conv.implied_prob,
                })

        df = pd.DataFrame(rows)
        if len(df) > 0:
            df = df.sort_values('decimal', ascending=False)

        return df

    def arbitrage_opportunity(self) -> Optional[Dict]:
        """
        Check for arbitrage opportunities.

        Returns details if arbitrage exists, None otherwise.
        """
        outcomes = set()
        for book_outcomes in self.odds_by_book.values():
            outcomes.update(book_outcomes.keys())

        if len(outcomes) != 2:
            return None  # Only handles 2-way markets

        outcomes = list(outcomes)
        best_1 = self.best_odds(outcomes[0])
        best_2 = self.best_odds(outcomes[1])

        if best_1[1] is None or best_2[1] is None:
            return None

        total_implied = best_1[1].implied_prob + best_2[1].implied_prob

        if total_implied < 1.0:
            profit_pct = (1 / total_implied - 1) * 100

            # Calculate stakes for equal profit
            stake_1 = best_2[1].implied_prob / total_implied
            stake_2 = best_1[1].implied_prob / total_implied

            return {
                'is_arbitrage': True,
                'profit_percent': profit_pct,
                'total_implied': total_implied,
                outcomes[0]: {
                    'book': best_1[0],
                    'odds': best_1[1],
                    'stake_fraction': stake_1,
                },
                outcomes[1]: {
                    'book': best_2[0],
                    'odds': best_2[1],
                    'stake_fraction': stake_2,
                },
            }

        return None


def calculate_parlay_odds(
    odds_list: List[float],
    odds_format: str = 'decimal'
) -> float:
    """
    Calculate parlay odds from individual legs.

    Args:
        odds_list: List of odds for each leg
        odds_format: Format of input odds

    Returns:
        Combined parlay odds in decimal format
    """
    decimal_odds = []
    for odds in odds_list:
        if odds_format == 'american':
            decimal_odds.append(american_to_decimal(odds))
        else:
            decimal_odds.append(odds)

    return np.prod(decimal_odds)


def parlay_ev(
    probabilities: List[float],
    odds_list: List[float],
    odds_format: str = 'decimal'
) -> Dict:
    """
    Calculate expected value of a parlay.

    Args:
        probabilities: Win probability for each leg
        odds_list: Odds for each leg

    Returns:
        Dictionary with EV analysis
    """
    parlay_prob = np.prod(probabilities)
    parlay_odds = calculate_parlay_odds(odds_list, odds_format)

    ev = calculate_ev(parlay_prob, parlay_odds, 'decimal')

    return {
        'parlay_probability': parlay_prob,
        'parlay_odds_decimal': parlay_odds,
        'parlay_odds_american': decimal_to_american(parlay_odds),
        'expected_value': ev,
        'ev_percent': ev * 100,
        'is_positive_ev': ev > 0,
    }
