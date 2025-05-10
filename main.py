import argparse
from markov import MarkovChain
from fuzzy import get_fast_attractiveness, attractiveness_ctrl
import numpy as np
import skfuzzy.control as ctrl
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

def classify_return(r, threshold=0.02):
	if r > threshold:
		return "Growth"
	elif r < -threshold:
		return "Decline"
	else:
		return "Stable"

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

def estimate_returns(prices, state_sequence):
	returns = prices.pct_change().dropna()
	grouped_returns = {"Growth": [], "Stable": [], "Decline": []}

	for r, s in zip(returns, state_sequence):
		grouped_returns[s].append(r)

	return {
		state: np.mean(grouped_returns[state])
		for state in grouped_returns
	}

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

def simulate_fuzzy_strategy(simulator, start_state, market_score, vol_score, returns, market_chain, days):
	state = start_state
	total_return = 1.0

	for _ in range(days):
		attractiveness = get_fast_attractiveness(simulator, market_score, vol_score)
		if attractiveness > 50:
			r = returns[state]
			total_return *= (1 + r)
		state = market_chain.next_state(state)

	return total_return

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", choices=["forecast", "backtest"], default="forecast")
	parser.add_argument("--ticker", type=str, default="AAPL")
	args = parser.parse_args()

	ticker = args.ticker
	mode = args.mode
	today = datetime.today()

	if mode == "forecast":
		start_date = today - timedelta(days=730)
		data = yf.download(ticker, start=start_date, end=today, progress=False, auto_adjust=True)
	else:  # backtest
		start_date = datetime(today.year - 2, 1, 1)
		end_date = datetime(today.year - 1, 1, 1)
		eval_end = datetime(today.year - 1, 12, 31)
		data = yf.download(ticker, start=start_date, end=eval_end, progress=False, auto_adjust=True)

	if isinstance(data.columns, pd.MultiIndex):
		prices = data["Close", ticker]
	else:
		prices = data["Close"]

	prices = pd.to_numeric(prices, errors="coerce").dropna()

	transition_matrix, state_sequence = build_transition_matrix(prices)
	returns = estimate_returns(prices, state_sequence)

	if mode == "forecast":
		market_score, vol_score, initial_state = get_market_features(prices)
	else:
		past_year_prices = prices[(prices.index >= pd.Timestamp(end_date)) & (prices.index <= pd.Timestamp(eval_end))]
		market_score, vol_score, initial_state = get_market_features(past_year_prices)

	print(f"Mode: {mode}")
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
	days = 252

	simulators = [ctrl.ControlSystemSimulation(attractiveness_ctrl) for _ in range(sim_count)]
	fuzzy_results = [
		simulate_fuzzy_strategy(simulators[i], initial_state, market_score, vol_score, returns, market_chain, days)
		for i in range(sim_count)
	]
	avg_fuzzy = sum(fuzzy_results) / sim_count

	print(f"\nFuzzy strategy: average final capital after {sim_count} simulations = {avg_fuzzy:.4f}")

	if mode == "backtest":
		actual_prices = prices[(prices.index >= pd.Timestamp(end_date)) & (prices.index <= pd.Timestamp(eval_end))]
		if not actual_prices.empty:
			actual_return = actual_prices.iloc[-1] / actual_prices.iloc[0]
			print(f"Real return from {actual_prices.index[0].date()} to {actual_prices.index[-1].date()} = {actual_return:.4f}")
			diff = avg_fuzzy - actual_return
			print(f"Difference between simulated and real return = {diff:.4f}")
		else:
			print("Not enough real price data for comparison.")
	else:
		if avg_fuzzy > 1.0:
			print("Recommendation: Fuzzy-based strategy appears profitable.")
		else:
			print("Recommendation: Strategy underperforms or is neutral.")
