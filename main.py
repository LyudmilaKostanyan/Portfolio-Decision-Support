import argparse
import yfinance as yf
import pandas as pd
import numpy as np
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt  # Added for visualization

from datetime import datetime
from markov import MarkovChain
from fuzzy import get_fast_attractiveness, attractiveness_ctrl


def classify_return(r: float, volatility: float, scale: float = 1.5) -> str:
	"""
	Classify a daily return into one of three market states based on adaptive threshold:
	  - 'Growth'   if return > scale × volatility
	  - 'Decline'  if return < -scale × volatility
	  - 'Stable'   otherwise
	"""
	threshold = scale * volatility
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
	std_dev = window.pct_change().std()
	# Modified: Reduced scaling factor from 1000 to 500
	volatility_score = min(max(std_dev * 500, 0), 100)
	diff = (window.iloc[-1] - window.mean()) / window.mean()

	threshold = std_dev  # adaptive threshold
	if diff > threshold:
		return 80.0, volatility_score, "Growth"
	elif diff < -threshold:
		return 20.0, volatility_score, "Decline"
	else:
		return 50.0, volatility_score, "Stable"

def simulate_fis_on_real_data(prices: pd.Series) -> float:
	capital = 1.0
	position = 0  # 0 = cash, 1 = long
	trades = 0  # Track number of trades
	positions = []  # For visualization
	attractiveness_scores = []  # For visualization
	price_history = []  # For visualization

	returns = prices.pct_change().dropna().reset_index(drop=True)
	prices = prices.iloc[1:].reset_index(drop=True)

	for i in range(30, len(prices) - 1):
		window = prices[i-30 : i]
		std_dev = window.pct_change().std()
		diff_ratio = (window.iloc[-1] - window.mean()) / window.mean()

		if diff_ratio > std_dev:
			market_state_val = 80.0
		elif diff_ratio < -std_dev:
			market_state_val = 20.0
		else:
			market_state_val = 50.0

		# Modified: Reduced scaling factor from 1000 to 500
		volatility_val = min(max(std_dev * 500, 0), 100)

		simulator = ctrl.ControlSystemSimulation(attractiveness_ctrl)
		attractiveness = get_fast_attractiveness(simulator, market_state_val, volatility_val)
		ret = returns[i + 1]

		old_position = position
		# Modified: Lowered buy threshold to 50 and sell threshold to 30
		if attractiveness > 50 and position == 0:
			position = 1
			trades += 1
		elif attractiveness < 30 and position == 1:
			position = 0
			trades += 1

		if position == 1:
			capital *= (1 + ret)

		# Log state for debugging
		print(f"Day {i}: Price={prices[i]:.2f}, MarketState={market_state_val}, "
		      f"Volatility={volatility_val:.2f}, Attractiveness={attractiveness:.2f}, "
		      f"Position={position}")
		positions.append(position)
		attractiveness_scores.append(attractiveness)
		price_history.append(prices[i])

	print(f"Total trades: {trades}")
	# Added: Visualize trading decisions
	plt.figure(figsize=(12, 6))
	plt.subplot(3, 1, 1)
	plt.plot(price_history, label="Price")
	plt.title("Price History")
	plt.legend()
	plt.subplot(3, 1, 2)
	plt.plot(attractiveness_scores, label="Attractiveness", color="orange")
	plt.axhline(y=50, color="g", linestyle="--", label="Buy Threshold")
	plt.axhline(y=30, color="r", linestyle="--", label="Sell Threshold")
	plt.title("Attractiveness Scores")
	plt.legend()
	plt.subplot(3, 1, 3)
	plt.plot(positions, label="Position", color="purple")
	plt.title("Position (0=Cash, 1=Long)")
	plt.legend()
	plt.tight_layout()
	plt.show()

	return capital

def simulate_markov_hold(
	start_state: str,
	transition_matrix: list[list[float]],
	state_returns: dict[str, float],
	days: int = 252,
	n: int = 1000
) -> float:
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

	data = yf.download(
		args.ticker,
		period="1y",
		progress=False,
		auto_adjust=True
	)

	prices = data["Close"]
	if isinstance(prices, pd.DataFrame):
		prices = prices.iloc[:, 0]
	prices = prices.dropna()

	returns = prices.pct_change().dropna()

	# Compute rolling volatility for adaptive classification
	rolling_std = returns.rolling(window=30).std().fillna(method="bfill")
	states = [classify_return(r, sigma) for r, sigma in zip(returns, rolling_std)]

	transition_matrix = build_transition_matrix(states)
	state_returns = estimate_state_returns(returns, states)

	_, _, current_state = get_market_features(prices[-30:])

	if args.mode == "backtest":
		real_pct = prices.iloc[-1] / prices.iloc[0] - 1.0

		fis_mul = simulate_fis_on_real_data(prices)
		markov_mul = simulate_markov_hold(current_state, transition_matrix, state_returns)

		initial_capital = 1000.0
		real_final = initial_capital * (1 + real_pct)
		real_profit = real_final - initial_capital

		fis_final = initial_capital * fis_mul
		fis_profit = fis_final - initial_capital

		markov_final = initial_capital * markov_mul
		markov_profit = markov_final - initial_capital

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
		fis_mul = simulate_fis_on_real_data(prices)
		markov_mul = simulate_markov_hold(current_state, transition_matrix, state_returns)

		print("\nForecast for the next trading year (capital multipliers):")
		print(f"\t• FIS strategy:    {fis_mul:.4f}×")
		print(f"\t• Markov-hold:     {markov_mul:.4f}×\n")

if __name__ == "__main__":
	main()