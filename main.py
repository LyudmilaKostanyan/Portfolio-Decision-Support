import argparse
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from markov import MarkovChain
from fuzzy import get_fast_attractiveness, attractiveness_ctrl
import skfuzzy.control as ctrl


def classify_return(r, threshold=0.02):
	if r > threshold:
		return "Growth"
	elif r < -threshold:
		return "Decline"
	else:
		return "Stable"


def build_transition_matrix(states):
	transitions = {s: {t: 0 for t in ["Growth", "Stable", "Decline"]} for s in ["Growth", "Stable", "Decline"]}
	for i in range(1, len(states)):
		prev, curr = states[i - 1], states[i]
		transitions[prev][curr] += 1

	matrix = []
	for s in ["Growth", "Stable", "Decline"]:
		total = sum(transitions[s].values())
		if total == 0:
			matrix.append([1 / 3] * 3)
		else:
			matrix.append([transitions[s][t] / total for t in ["Growth", "Stable", "Decline"]])
	return matrix


def estimate_state_returns(returns, states):
	grouped = {s: [] for s in ["Growth", "Stable", "Decline"]}
	for r, s in zip(returns, states):
		grouped[s].append(r)
	return {s: np.mean(grouped[s]) if grouped[s] else 0.0 for s in grouped}


def get_market_features(window):
	vol = min(max(window.pct_change().std() * 1000, 0), 100)
	diff = (window.iloc[-1] - window.mean()) / window.mean()
	if diff > 0.02:
		state = "Growth"
		score = 80
	elif diff < -0.02:
		state = "Decline"
		score = 20
	else:
		state = "Stable"
		score = 50
	return score, vol, state


def simulate_fis_on_real_data(prices):
	total_return = 1.0
	for i in range(30, len(prices) - 1):
		window = prices[i - 30:i]
		market_score, vol_score, state = get_market_features(window)
		attractiveness = get_fast_attractiveness(ctrl.ControlSystemSimulation(attractiveness_ctrl), market_score, vol_score)
		ret = prices.pct_change().iloc[i + 1]
		if attractiveness > 60:
			total_return *= (1 + ret)
		elif attractiveness < 40:
			total_return *= (1 - ret)
	return total_return


def simulate_markov_hold(start_state, transition_matrix, state_returns, days=252, n=1000):
	mc = MarkovChain(["Growth", "Stable", "Decline"], transition_matrix)
	results = []
	for _ in range(n):
		state = start_state
		capital = 1.0
		for _ in range(days):
			capital *= (1 + state_returns[state])
			state = mc.next_state(state)
		results.append(capital)
	return sum(results) / n


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--ticker", default="AAPL")
	parser.add_argument("--mode", choices=["backtest", "forecast"], default="backtest")
	args = parser.parse_args()

	today = datetime.today()
	if args.mode == "backtest":
		start = f"{today.year - 2}-01-01"
		end = f"{today.year - 1}-12-31"
	else:
		start = f"{today.year - 2}-01-01"
		end = today.strftime("%Y-%m-%d")

	data = yf.download(args.ticker, start=start, end=end, progress=False, auto_adjust=True)

	# FIX: safely extract Close prices
	if isinstance(data.columns, pd.MultiIndex):
		prices = data["Close", args.ticker]
	else:
		prices = data["Close"]

	prices = pd.to_numeric(prices, errors="coerce").dropna()
	returns = prices.pct_change().dropna()
	states = [classify_return(r) for r in returns]

	transition_matrix = build_transition_matrix(states)
	state_returns = estimate_state_returns(returns, states)
	market_score, vol_score, init_state = get_market_features(prices[-30:])

	if args.mode == "backtest":
		real_return = prices.iloc[-1] / prices.iloc[0]
		fis_result = simulate_fis_on_real_data(prices)
		markov_result = simulate_markov_hold(init_state, transition_matrix, state_returns)
		print(f"Real market return: {real_return:.4f}")
		print(f"FIS strategy return:  {fis_result:.4f}")
		print(f"Markov hold return:  {markov_result:.4f}")
	else:
		fis_sim = simulate_fis_on_real_data(prices)
		markov_sim = simulate_markov_hold(init_state, transition_matrix, state_returns)
		print(f"Forecast (simulated future)")
		print(f"FIS strategy capital:   {fis_sim:.4f}")
		print(f"Markov hold capital:    {markov_sim:.4f}")


if __name__ == "__main__":
	main()
