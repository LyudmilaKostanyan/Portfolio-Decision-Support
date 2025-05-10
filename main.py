from markov import MarkovChain
from fuzzy import get_fast_attractiveness, attractiveness_ctrl
from portfolio import classify_state, build_transition_matrix, simulate_strategy
import yfinance as yf
import pandas as pd
import numpy as np
import skfuzzy.control as ctrl

if __name__ == "__main__":
	# Parameters
	ticker = 'AAPL'
	period = '1y'
	state_labels = ['Growth', 'Stable', 'Decline']
	state_score_map = {'Growth': 80, 'Stable': 50, 'Decline': 20}
	returns_map = {'Growth': 0.05, 'Stable': 0.0, 'Decline': -0.04}
	days = 100
	sim_count = 1000

	# Step 1: Load data and calculate return + volatility
	df = yf.download(ticker, period=period, auto_adjust=False)
	df['Return'] = df['Close'].pct_change()
	df['Volatility'] = df['Return'].rolling(5).std() * 100
	df.dropna(inplace=True)

	# Step 2: Classify market states
	df['State'] = df['Return'].apply(classify_state)
	state_sequence = df['State'].tolist()
	volatility_samples = df['Volatility'].values

	# Step 3: Build Markov Chain
	transition_matrix = build_transition_matrix(state_sequence, state_labels)
	mc = MarkovChain(state_labels, transition_matrix)

	# Step 4: Run simulations
	results = [
		simulate_strategy('Stable', mc, volatility_samples, state_score_map, returns_map, days)
		for _ in range(sim_count)
	]

	# Step 5: Print result
	avg = sum(results) / sim_count
	print(f"Avg capital after {sim_count} simulations: {avg:.4f}")
