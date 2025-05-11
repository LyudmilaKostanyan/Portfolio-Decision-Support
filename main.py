import argparse
import yfinance as yf
import pandas as pd
import numpy as np
import skfuzzy.control as ctrl

from datetime import datetime
from markov import MarkovChain
from fuzzy import get_fast_attractiveness, attractiveness_ctrl

def classify_return(r: float, threshold: float = 0.02) -> str:
	"""
	Classify a daily return into one of three market states:
	  - 'Growth'   if return > threshold
	  - 'Decline'  if return < -threshold
	  - 'Stable'   otherwise
	"""
	if r > threshold:
		return "Growth"
	elif r < -threshold:
		return "Decline"
	else:
		return "Stable"

def build_transition_matrix(states: list[str]) -> list[list[float]]:
	"""
	Count observed state transitions and normalize into probabilities.
	Returns a 3×3 matrix for states ['Growth', 'Stable', 'Decline'].
	"""
	categories = ["Growth", "Stable", "Decline"]
	counts = {s: {t: 0 for t in categories} for s in categories}
	for prev, curr in zip(states, states[1:]):
		counts[prev][curr] += 1

	matrix: list[list[float]] = []
	for s in categories:
		total = sum(counts[s].values())
		if total == 0:
			matrix.append([1/3, 1/3, 1/3])
		else:
			matrix.append([counts[s][t] / total for t in categories])
	return matrix

def estimate_state_returns(returns: pd.Series, states: list[str]) -> dict[str, float]:
	"""
	Compute the average daily return observed in each market state.
	"""
	grouped = {"Growth": [], "Stable": [], "Decline": []}
	for r, s in zip(returns, states):
		grouped[s].append(r)
	return {s: (np.mean(vals) if vals else 0.0) for s, vals in grouped.items()}

def get_market_features(window: pd.Series) -> tuple[float, float, str]:
	"""
	From the last 30 prices:
	  - volatility_score: scaled standard deviation (0–100)
	  - state: based on deviation from 30-day mean
	Returns (market_score, volatility_score, state).
	"""
	volatility_score = min(max(window.pct_change().std() * 1000, 0), 100)
	diff = (window.iloc[-1] - window.mean()) / window.mean()
	if diff > 0.02:
		return 80.0, volatility_score, "Growth"
	elif diff < -0.02:
		return 20.0, volatility_score, "Decline"
	else:
		return 50.0, volatility_score, "Stable"

def simulate_fis_on_real_data(prices: pd.Series) -> float:
	capital = 1.0
	position = 0  # 0 = cash, 1 = long

	# daily returns and aligned price series
	returns = prices.pct_change().dropna().reset_index(drop=True)
	prices = prices.iloc[1:].reset_index(drop=True)

	for i in range(30, len(prices) - 1):
		window = prices[i-30 : i]

		# 1) compute market_state_val (0–100 scale)
		diff_ratio = (window.iloc[-1] - window.mean()) / window.mean()
		if diff_ratio > 0.02:
			market_state_val = 80.0
		elif diff_ratio < -0.02:
			market_state_val = 20.0
		else:
			market_state_val = 50.0

		# 2) compute volatility_val (std dev scaled to 0–100)
		volatility_val = min(max(window.pct_change().std() * 1000, 0), 100)

		# 3) evaluate FIS
		simulator = ctrl.ControlSystemSimulation(attractiveness_ctrl)
		attractiveness = get_fast_attractiveness(simulator, market_state_val, volatility_val)
		ret = returns[i + 1]

		# 4) entry/exit logic
		if attractiveness > 60 and position == 0:
			position = 1  # enter long
		elif attractiveness < 40 and position == 1:
			position = 0  # exit to cash

		# 5) apply P/L only when long
		if position == 1:
			capital *= (1 + ret)

	return capital

def simulate_markov_hold(
	start_state: str,
	transition_matrix: list[list[float]],
	state_returns: dict[str, float],
	days: int = 252,
	n: int = 1000
) -> float:
	"""
	Monte Carlo simulation of a Markov-hold strategy:
	  - start in start_state
	  - for `days` trading days, multiply capital by average return for current state
	  - randomly transition to next state
	Returns average end-of-period multiplier over `n` runs.
	"""
	mc = MarkovChain(["Growth", "Stable", "Decline"], transition_matrix)
	results: list[float] = []
	for _ in range(n):
		state = start_state
		capital = 1.0
		for _ in range(days):
			capital *= (1 + state_returns[state])
			state = mc.next_state(state)
		results.append(capital)
	return sum(results) / n

def main():
	parser = argparse.ArgumentParser(
		description="Compare real NVDA performance with FIS and Markov strategies"
	)
	parser.add_argument(
		"--ticker", default="NVDA",
		help="Ticker symbol, e.g. NVDA"
	)
	parser.add_argument(
		"--mode", choices=["backtest", "forecast"],
		default="backtest",
		help="backtest: past year; forecast: simulate next year"
	)
	args = parser.parse_args()

	# Download one year of adjusted closing prices
	data = yf.download(
		args.ticker,
		period="1y",
		progress=False,
		auto_adjust=True
	)

	# Extract the Close price series (ensure it's a Series, not DataFrame)
	prices = data["Close"]
	if isinstance(prices, pd.DataFrame):
		prices = prices.iloc[:, 0]
	prices = prices.dropna()

	# Compute daily returns and classify market states
	returns = prices.pct_change().dropna()
	states = [classify_return(r) for r in returns]

	# Build Markov transition matrix and average returns per state
	transition_matrix = build_transition_matrix(states)
	state_returns = estimate_state_returns(returns, states)

	# Determine current market state from last 30 days
	_, _, current_state = get_market_features(prices[-30:])

	if args.mode == "backtest":
		# Real percent return over the year
		real_pct = prices.iloc[-1] / prices.iloc[0] - 1.0

		# Strategy multipliers
		fis_mul = simulate_fis_on_real_data(prices)
		markov_mul = simulate_markov_hold(current_state, transition_matrix, state_returns)

		# Hypothetical $1,000 investment
		initial_capital = 1000.0
		real_final = initial_capital * (1 + real_pct)
		real_profit = real_final - initial_capital

		fis_final = initial_capital * fis_mul
		fis_profit = fis_final - initial_capital

		markov_final = initial_capital * markov_mul
		markov_profit = markov_final - initial_capital

		# Plain-language output
		print(f"\nIf you had invested $1,000 in {args.ticker} one year ago:")
		print(f"\t• Today it would be worth ${real_final:,.2f},")
		print(f"\t  a profit of ${real_profit:,.2f} ({real_pct:.2%}).\n")

		print("Under the FIS-based trading strategy:")
		print(f"\t• Your capital would be ${fis_final:,.2f},")
		print(f"\t  a profit of ${fis_profit:,.2f} ({fis_mul - 1:.2%}).\n")

		print("Under the Markov-hold strategy:")
		print(f"\t• Your capital would be ${markov_final:,.2f},")
		print(f"\t  a profit of ${markov_profit:,.2f} ({markov_mul - 1:.2%}).\n")
	else:
		# Forecast mode: show multipliers for next year
		fis_mul = simulate_fis_on_real_data(prices)
		markov_mul = simulate_markov_hold(current_state, transition_matrix, state_returns)

		print("\nForecast for the next trading year (capital multipliers):")
		print(f"\t• FIS strategy:    {fis_mul:.4f}×")
		print(f"\t• Markov-hold:     {markov_mul:.4f}×\n")

if __name__ == "__main__":
	main()
