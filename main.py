from markov import MarkovChain
from fuzzy import get_fast_attractiveness, attractiveness_ctrl
import numpy as np
import skfuzzy.control as ctrl
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

# 1. Classify a return value into one of the market states
def classify_return(r, threshold=0.02):
	if r > threshold:
		return "Growth"
	elif r < -threshold:
		return "Decline"
	else:
		return "Stable"

# 2. Build the transition matrix based on historical classified states
def build_transition_matrix(prices, threshold=0.02):
	returns = prices.pct_change().dropna()
	states = [classify_return(r, threshold) for r in returns]

	transitions = {"Growth": {"Growth": 0, "Stable": 0, "Decline": 0},
	               "Stable": {"Growth": 0, "Stable": 0, "Decline": 0},
	               "Decline": {"Growth": 0, "Stable": 0, "Decline": 0}}

	for i in range(1, len(states)):
		prev, curr = states[i - 1], states[i]
		transitions[prev][curr] += 1

	matrix = []
	for from_state in ["Growth", "Stable", "Decline"]:
		total = sum(transitions[from_state].values())
		if total == 0:
			row = [1/3, 1/3, 1/3]
		else:
			row = [transitions[from_state][to_state] / total for to_state in ["Growth", "Stable", "Decline"]]
		matrix.append(row)

	return matrix, states

# 3. Estimate average return for each market state
def estimate_returns(prices, state_sequence):
	returns = prices.pct_change().dropna()
	grouped_returns = {"Growth": [], "Stable": [], "Decline": []}

	for r, s in zip(returns, state_sequence):
		grouped_returns[s].append(r)

	return {
		state: np.mean(grouped_returns[state])
		for state in grouped_returns
	}

# 4. Evaluate current market state score and volatility score
def get_market_features(prices, window=30):
	returns = prices.pct_change().dropna()
	recent_returns = returns[-window:]
	volatility_score = min(max(recent_returns.std() * 1000, 0), 100)

	recent_prices = prices[-window:]
	mean_price = recent_prices.mean()
	last_price = recent_prices.iloc[-1]
	diff_ratio = (last_price - mean_price) / mean_price

	if diff_ratio > 0.02:
		market_state_score = 80
		initial_state = 'Growth'
	elif diff_ratio < -0.02:
		market_state_score = 20
		initial_state = 'Decline'
	else:
		market_state_score = 50
		initial_state = 'Stable'

	return market_state_score, volatility_score, initial_state

if __name__ == "__main__":
	ticker = "AAPL"
	today = datetime.today()
	start_date = today - timedelta(days=730)

	# Download historical market data (auto_adjust is True by default)
	data = yf.download(ticker, start=start_date, end=today, progress=False, auto_adjust=True)

	# Extract 'Close' column safely regardless of column format
	if isinstance(data.columns, pd.MultiIndex):
		prices = data["Close", ticker]
	else:
		prices = data["Close"]

	# Convert to numeric and drop NaN
	prices = pd.to_numeric(prices, errors="coerce").dropna()

	# Derive parameters from the market
	transition_matrix, state_sequence = build_transition_matrix(prices)
	returns = estimate_returns(prices, state_sequence)
	market_score, vol_score, initial_state = get_market_features(prices)

	print(f"Ticker: {ticker}")
	print("Transition Matrix:")
	print(np.round(transition_matrix, 3))
	print("Estimated Returns per State:")
	for k, v in returns.items():
		print(f"  {k}: {v:.4f}")
	print(f"Market state score: {market_score}")
	print(f"Volatility score: {vol_score:.2f}")
	print(f"Initial market state: {initial_state}")

	states = ['Growth', 'Stable', 'Decline']
	market_chain = MarkovChain(states, transition_matrix)

	sim_count = 1000
	days = 252  # ~1 trading year

	simulators = [ctrl.ControlSystemSimulation(attractiveness_ctrl) for _ in range(sim_count)]

	def simulate_fuzzy_strategy(simulator, start_state):
		state = start_state
		total_return = 1.0

		for _ in range(days):
			attractiveness = get_fast_attractiveness(simulator, market_score, vol_score)
			if attractiveness > 50:
				r = returns[state]
				total_return *= (1 + r)
			state = market_chain.next_state(state)

		return total_return

	fuzzy_results = [
		simulate_fuzzy_strategy(simulators[i], initial_state)
		for i in range(sim_count)
	]

	avg_fuzzy = sum(fuzzy_results) / sim_count

	print(f"\nFuzzy strategy: average final capital after {sim_count} simulations = {avg_fuzzy:.4f}")
	if avg_fuzzy > 1.0:
		print("Recommendation: Fuzzy-based strategy appears profitable.")
	else:
		print("Recommendation: Strategy underperforms or is neutral.")
