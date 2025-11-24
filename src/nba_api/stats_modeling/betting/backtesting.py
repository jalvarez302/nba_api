"""
Backtesting Framework for Sports Betting Strategies

Professional backtesting with:
- Walk-forward validation
- Realistic simulation (vig, line movement)
- Comprehensive performance metrics
- Risk-adjusted returns
- Monte Carlo analysis
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.base import clone
import warnings


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    # Core metrics
    total_bets: int
    wins: int
    losses: int
    pushes: int
    win_rate: float
    total_wagered: float
    total_profit: float
    roi: float

    # Risk metrics
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Betting metrics
    avg_odds: float
    avg_edge: float
    clv: float  # Closing Line Value
    yield_per_bet: float

    # Time series
    bankroll_history: List[float]
    profit_history: List[float]
    bet_details: pd.DataFrame

    # Metadata
    start_date: str
    end_date: str
    strategy_name: str
    params: Dict

    def summary(self) -> str:
        """Generate summary string."""
        return f"""
Backtest Results: {self.strategy_name}
{'='*50}
Period: {self.start_date} to {self.end_date}

PERFORMANCE
-----------
Total Bets: {self.total_bets}
Record: {self.wins}-{self.losses}-{self.pushes}
Win Rate: {self.win_rate:.1%}
ROI: {self.roi:.2%}
Total Profit: ${self.total_profit:,.2f}

RISK METRICS
------------
Max Drawdown: {self.max_drawdown:.1%}
Sharpe Ratio: {self.sharpe_ratio:.2f}
Sortino Ratio: {self.sortino_ratio:.2f}

BETTING METRICS
---------------
Avg Odds: {self.avg_odds:.3f}
Avg Edge: {self.avg_edge:.1%}
CLV: {self.clv:.1%}
Yield/Bet: {self.yield_per_bet:.2%}
"""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'total_bets': self.total_bets,
            'wins': self.wins,
            'losses': self.losses,
            'pushes': self.pushes,
            'win_rate': self.win_rate,
            'total_wagered': self.total_wagered,
            'total_profit': self.total_profit,
            'roi': self.roi,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'avg_odds': self.avg_odds,
            'avg_edge': self.avg_edge,
            'clv': self.clv,
            'yield_per_bet': self.yield_per_bet,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'strategy_name': self.strategy_name,
        }


class Backtester:
    """
    Professional backtesting framework for betting strategies.

    Features:
    - Walk-forward model training
    - Realistic simulation with vig and line movement
    - Comprehensive performance metrics
    - Monte Carlo analysis
    """

    def __init__(
        self,
        initial_bankroll: float = 10000,
        bet_sizing: str = 'fractional_kelly',
        kelly_fraction: float = 0.25,
        max_bet_pct: float = 0.05,
        min_bet_pct: float = 0.01,
        vig: float = 0.05,  # Assumed bookmaker vig
    ):
        """
        Initialize backtester.

        Args:
            initial_bankroll: Starting bankroll
            bet_sizing: 'kelly', 'fractional_kelly', 'fixed', or 'unit'
            kelly_fraction: Fraction of Kelly to use
            max_bet_pct: Maximum bet as % of bankroll
            min_bet_pct: Minimum bet as % of bankroll
            vig: Assumed bookmaker vig
        """
        self.initial_bankroll = initial_bankroll
        self.bet_sizing = bet_sizing
        self.kelly_fraction = kelly_fraction
        self.max_bet_pct = max_bet_pct
        self.min_bet_pct = min_bet_pct
        self.vig = vig

    def backtest_props(
        self,
        model,
        data: pd.DataFrame,
        lines: Dict[Tuple[int, str], float],  # (player_id, game_id) -> line
        actuals: Dict[Tuple[int, str], float],  # (player_id, game_id) -> actual
        min_edge: float = 0.03,
        train_window: int = 500,
        retrain_frequency: int = 50,
        strategy_name: str = 'Props Strategy'
    ) -> BacktestResult:
        """
        Backtest player props strategy.

        Args:
            model: Props prediction model (must have fit/predict methods)
            data: DataFrame with player game logs
            lines: Dictionary of betting lines
            actuals: Dictionary of actual results
            min_edge: Minimum edge to place bet
            train_window: Number of samples for training
            retrain_frequency: How often to retrain model

        Returns:
            BacktestResult with performance metrics
        """
        data = data.sort_values('GAME_DATE').reset_index(drop=True)

        bankroll = self.initial_bankroll
        bankroll_history = [bankroll]
        bet_details = []

        wins, losses, pushes = 0, 0, 0
        total_wagered = 0

        for i in range(train_window, len(data), retrain_frequency):
            # Train on historical data
            train_data = data.iloc[max(0, i-train_window):i]
            test_data = data.iloc[i:min(i+retrain_frequency, len(data))]

            if len(train_data) < 100:
                continue

            # Fit model
            try:
                model_copy = clone(model) if hasattr(model, '__sklearn_clone__') else model
                model_copy.fit(train_data)
            except Exception as e:
                warnings.warn(f"Training failed at step {i}: {e}")
                continue

            # Make predictions on test data
            for _, row in test_data.iterrows():
                player_id = row.get('PLAYER_ID')
                game_id = row.get('GAME_ID')
                key = (player_id, game_id)

                if key not in lines or key not in actuals:
                    continue

                line = lines[key]
                actual = actuals[key]

                try:
                    # Get prediction
                    pred = model_copy.predict_line_probability(
                        pd.DataFrame([row]),
                        line=line,
                        over=True,
                        engineer_features=True
                    )[0]

                    edge_over = pred - 0.5
                    edge_under = (1 - pred) - 0.5

                    # Decide bet direction
                    if edge_over >= min_edge:
                        bet_direction = 'over'
                        win_prob = pred
                        edge = edge_over
                    elif edge_under >= min_edge:
                        bet_direction = 'under'
                        win_prob = 1 - pred
                        edge = edge_under
                    else:
                        continue  # No edge

                    # Calculate odds (with vig)
                    fair_odds = 1 / win_prob
                    book_odds = fair_odds * (1 - self.vig)

                    # Calculate bet size
                    stake = self._calculate_stake(bankroll, win_prob, book_odds)

                    if stake < 1:  # Minimum bet
                        continue

                    # Determine result
                    if bet_direction == 'over':
                        won = actual > line
                    else:
                        won = actual < line

                    # Handle push (exact line)
                    if actual == line:
                        profit = 0
                        pushes += 1
                        result = 'push'
                    elif won:
                        profit = stake * (book_odds - 1)
                        wins += 1
                        result = 'win'
                    else:
                        profit = -stake
                        losses += 1
                        result = 'loss'

                    bankroll += profit
                    total_wagered += stake
                    bankroll_history.append(bankroll)

                    bet_details.append({
                        'date': row.get('GAME_DATE'),
                        'player_id': player_id,
                        'line': line,
                        'actual': actual,
                        'direction': bet_direction,
                        'predicted_prob': win_prob,
                        'edge': edge,
                        'odds': book_odds,
                        'stake': stake,
                        'profit': profit,
                        'result': result,
                        'bankroll': bankroll,
                    })

                except Exception:
                    continue

        # Calculate metrics
        return self._calculate_results(
            bet_details, bankroll_history, strategy_name, data
        )

    def backtest_spreads(
        self,
        model,
        team_data: pd.DataFrame,
        spreads: Dict[str, float],  # game_id -> spread (home perspective)
        results: Dict[str, Tuple[int, int]],  # game_id -> (home_score, away_score)
        min_edge: float = 0.02,
        train_window: int = 200,
        retrain_frequency: int = 20,
        strategy_name: str = 'Spread Strategy'
    ) -> BacktestResult:
        """
        Backtest spread betting strategy.

        Args:
            model: Spread prediction model
            team_data: DataFrame with team game logs
            spreads: Dictionary of betting spreads
            results: Dictionary of game results
            min_edge: Minimum edge to place bet
        """
        # Similar structure to backtest_props
        # Implementation follows same pattern
        bankroll = self.initial_bankroll
        bankroll_history = [bankroll]
        bet_details = []
        wins, losses, pushes = 0, 0, 0
        total_wagered = 0

        # Get unique games sorted by date
        games = team_data.groupby('GAME_ID').first().reset_index()
        games = games.sort_values('GAME_DATE')

        for game_id in games['GAME_ID'].values:
            if game_id not in spreads or game_id not in results:
                continue

            spread = spreads[game_id]
            home_score, away_score = results[game_id]
            actual_margin = home_score - away_score

            # Get team data for this game
            game_data = team_data[team_data['GAME_ID'] == game_id]
            if len(game_data) < 2:
                continue

            try:
                # Predict spread
                home_data = game_data[game_data['IS_HOME'] == 1] if 'IS_HOME' in game_data.columns else game_data.iloc[:1]
                away_data = game_data[game_data['IS_HOME'] == 0] if 'IS_HOME' in game_data.columns else game_data.iloc[1:]

                pred_result = model.predict_spread_probability(
                    home_data, away_data, spread
                )

                home_cover_prob = pred_result['home_cover_probability']
                away_cover_prob = pred_result['away_cover_probability']

                # Determine bet
                edge_home = home_cover_prob - 0.5
                edge_away = away_cover_prob - 0.5

                if edge_home >= min_edge:
                    bet_side = 'home'
                    win_prob = home_cover_prob
                    edge = edge_home
                elif edge_away >= min_edge:
                    bet_side = 'away'
                    win_prob = away_cover_prob
                    edge = edge_away
                else:
                    continue

                # Standard -110 odds for spreads
                book_odds = 1.91

                stake = self._calculate_stake(bankroll, win_prob, book_odds)
                if stake < 1:
                    continue

                # Determine result
                if bet_side == 'home':
                    won = actual_margin > spread
                else:
                    won = actual_margin < spread

                if actual_margin == spread:
                    profit = 0
                    pushes += 1
                    result = 'push'
                elif won:
                    profit = stake * (book_odds - 1)
                    wins += 1
                    result = 'win'
                else:
                    profit = -stake
                    losses += 1
                    result = 'loss'

                bankroll += profit
                total_wagered += stake
                bankroll_history.append(bankroll)

                bet_details.append({
                    'game_id': game_id,
                    'spread': spread,
                    'actual_margin': actual_margin,
                    'bet_side': bet_side,
                    'predicted_prob': win_prob,
                    'edge': edge,
                    'odds': book_odds,
                    'stake': stake,
                    'profit': profit,
                    'result': result,
                    'bankroll': bankroll,
                })

            except Exception:
                continue

        return self._calculate_results(
            bet_details, bankroll_history, strategy_name, team_data
        )

    def _calculate_stake(
        self,
        bankroll: float,
        win_prob: float,
        odds: float
    ) -> float:
        """Calculate stake based on betting strategy."""
        from nba_api.stats_modeling.betting.bankroll import (
            kelly_criterion, fractional_kelly
        )

        if self.bet_sizing == 'kelly':
            pct = kelly_criterion(win_prob, odds)
        elif self.bet_sizing == 'fractional_kelly':
            pct = fractional_kelly(win_prob, odds, self.kelly_fraction)
        elif self.bet_sizing == 'fixed':
            pct = self.min_bet_pct
        else:
            pct = self.min_bet_pct

        pct = min(self.max_bet_pct, max(self.min_bet_pct, pct))
        return bankroll * pct

    def _calculate_results(
        self,
        bet_details: List[Dict],
        bankroll_history: List[float],
        strategy_name: str,
        data: pd.DataFrame
    ) -> BacktestResult:
        """Calculate comprehensive backtest results."""
        if not bet_details:
            return BacktestResult(
                total_bets=0, wins=0, losses=0, pushes=0,
                win_rate=0, total_wagered=0, total_profit=0, roi=0,
                max_drawdown=0, sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
                avg_odds=0, avg_edge=0, clv=0, yield_per_bet=0,
                bankroll_history=bankroll_history,
                profit_history=[0],
                bet_details=pd.DataFrame(),
                start_date='', end_date='',
                strategy_name=strategy_name, params={}
            )

        df = pd.DataFrame(bet_details)

        wins = len(df[df['result'] == 'win'])
        losses = len(df[df['result'] == 'loss'])
        pushes = len(df[df['result'] == 'push'])
        total_bets = len(df)

        total_wagered = df['stake'].sum()
        total_profit = df['profit'].sum()

        # Calculate returns for risk metrics
        returns = df['profit'] / df['stake']

        # Risk metrics
        max_dd = self._max_drawdown(bankroll_history)
        sharpe = self._sharpe_ratio(returns)
        sortino = self._sortino_ratio(returns)
        calmar = total_profit / max_dd if max_dd > 0 else 0

        # Dates
        if 'date' in df.columns:
            dates = pd.to_datetime(df['date'])
            start_date = dates.min().strftime('%Y-%m-%d') if len(dates) > 0 else ''
            end_date = dates.max().strftime('%Y-%m-%d') if len(dates) > 0 else ''
        else:
            start_date = ''
            end_date = ''

        return BacktestResult(
            total_bets=total_bets,
            wins=wins,
            losses=losses,
            pushes=pushes,
            win_rate=wins / total_bets if total_bets > 0 else 0,
            total_wagered=total_wagered,
            total_profit=total_profit,
            roi=total_profit / total_wagered if total_wagered > 0 else 0,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            avg_odds=df['odds'].mean(),
            avg_edge=df['edge'].mean(),
            clv=df['edge'].mean(),  # Simplified CLV
            yield_per_bet=total_profit / total_bets if total_bets > 0 else 0,
            bankroll_history=bankroll_history,
            profit_history=df['profit'].cumsum().tolist(),
            bet_details=df,
            start_date=start_date,
            end_date=end_date,
            strategy_name=strategy_name,
            params={
                'bet_sizing': self.bet_sizing,
                'kelly_fraction': self.kelly_fraction,
                'min_edge': 0.03,
            }
        )

    def _max_drawdown(self, bankroll_history: List[float]) -> float:
        """Calculate maximum drawdown."""
        if len(bankroll_history) < 2:
            return 0

        peak = bankroll_history[0]
        max_dd = 0

        for balance in bankroll_history:
            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak
            max_dd = max(max_dd, dd)

        return max_dd

    def _sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2 or returns.std() == 0:
            return 0
        return returns.mean() / returns.std() * np.sqrt(len(returns))

    def _sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio (only penalizes downside volatility)."""
        if len(returns) < 2:
            return 0

        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float('inf') if returns.mean() > 0 else 0

        return returns.mean() / downside_returns.std() * np.sqrt(len(returns))

    def monte_carlo_analysis(
        self,
        result: BacktestResult,
        n_simulations: int = 1000,
        n_bets: Optional[int] = None
    ) -> Dict:
        """
        Run Monte Carlo simulation on backtest results.

        Simulates future performance based on historical statistics.

        Returns:
            Dictionary with simulation results
        """
        if result.total_bets == 0:
            return {}

        n_bets = n_bets or result.total_bets

        # Parameters from backtest
        win_rate = result.win_rate
        avg_odds = result.avg_odds
        avg_stake_pct = 0.02  # Approximate

        final_bankrolls = []

        for _ in range(n_simulations):
            bankroll = self.initial_bankroll

            for _ in range(n_bets):
                stake = bankroll * avg_stake_pct

                if np.random.random() < win_rate:
                    profit = stake * (avg_odds - 1)
                else:
                    profit = -stake

                bankroll += profit

                if bankroll <= 0:
                    break

            final_bankrolls.append(bankroll)

        final_bankrolls = np.array(final_bankrolls)

        return {
            'median_final_bankroll': np.median(final_bankrolls),
            'mean_final_bankroll': np.mean(final_bankrolls),
            'std_final_bankroll': np.std(final_bankrolls),
            'percentile_5': np.percentile(final_bankrolls, 5),
            'percentile_25': np.percentile(final_bankrolls, 25),
            'percentile_75': np.percentile(final_bankrolls, 75),
            'percentile_95': np.percentile(final_bankrolls, 95),
            'probability_of_profit': (final_bankrolls > self.initial_bankroll).mean(),
            'probability_of_ruin': (final_bankrolls <= 0).mean(),
            'expected_roi': (np.mean(final_bankrolls) / self.initial_bankroll - 1),
        }
