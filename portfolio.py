import yfinance as yf
import pandas as pd
import numpy as np
from markov import MarkovChain
from fuzzy import get_fast_attractiveness, attractiveness_ctrl
import skfuzzy.control as ctrl

# Step 1 — classify return into market state
def classify_state(r):
	if r > 0.01:
		return 'Growth'
	elif r < -0.01:
		return 'Decline'
	else:
		return 'Stable'

# Step 2 — calculate transition matrix from state sequence
def build_transition_matrix(states, state_labels):
	n = len(state_labels)
	matrix = np.zeros((n, n))
	index = {label: i for i, label in enumerate(state_labels)}

	for (s1, s2) in zip(states[:-1], states[1:]):
		i, j = index[s1], index[s2]
		matrix[i][j] += 1

	# Normalize rows
	matrix = np.array([
		row / row.sum() if row.sum() > 0 else np.ones(n) / n
		for row in matrix
	])
	return matrix

# Step 3 — simulation using MarkovChain and FIS
def simulate_strategy(initial_state, markov_chain, volatility_series, state_score_map, returns_map, days):
	simulator = ctrl.ControlSystemSimulation(attractiveness_ctrl)
	state = initial_state
	total_return = 1.0

	for i in range(days):
		vol = np.random.choice(volatility_series)
		score = state_score_map[state]
		attractiveness = get_fast_attractiveness(simulator, score, vol)

		if attractiveness > 60:
			total_return *= (1 + returns_map[state])

		state = markov_chain.next_state(state)

	return total_return
