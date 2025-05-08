from markov import HiddenMarkovModel
from fuzzy import get_fast_attractiveness, attractiveness_ctrl
import numpy as np
import skfuzzy.control as ctrl
from collections import Counter

# Define synthetic return levels based on state index: 0 = Bullish, 1 = Neutral, 2 = Bearish
state_returns = {
	0: 0.05,	# Bullish
	1: 0.00,	# Neutral
	2: -0.04	# Bearish
}

# Map state index to fuzzy score for 'market_state' input
state_score_map = {
	0: 80,		# Bullish
	1: 50,		# Neutral
	2: 20		# Bearish
}

# Simulate one run using fuzzy logic and HMM
def simulate_fuzzy_strategy(simulator, hmm_model, observation_seq, hidden_states):
	total_return = 1.0
	for t in range(len(observation_seq)):
		volatility = np.random.uniform(0, 100)
		state = hidden_states[t]
		state_score = state_score_map[state]

		attractiveness = get_fast_attractiveness(simulator, state_score, volatility)

		if attractiveness > 60:
			r = state_returns[state]
			total_return *= (1 + r)

	return total_return

if __name__ == "__main__":
	sim_count = 1000
	days = 300

	# Step 1: Generate improved synthetic returns
	np.random.seed(42)
	bullish = np.random.normal(0.03, 0.01, size=(100, 1))    # stronger uptrend
	neutral = np.random.normal(0.0, 0.005, size=(100, 1))     # tighter neutral
	bearish = np.random.normal(-0.03, 0.015, size=(100, 1))   # stronger downtrend

	full_returns = np.concatenate([bullish, neutral, bearish])

	# Step 2: Train HMM
	hmm_model = HiddenMarkovModel()
	hmm_model.train(full_returns)
	hidden_states = hmm_model.predict_states(full_returns)

	# Print state distribution
	state_distribution = Counter(hidden_states)
	print("HMM state distribution:", state_distribution)

	# Optional: visualize hidden state segmentation
	hmm_model.plot_hidden_states(full_returns, hidden_states)

	# Step 3: Create FIS simulators
	simulators = [ctrl.ControlSystemSimulation(attractiveness_ctrl) for _ in range(sim_count)]

	# Step 4: Run simulations
	results = [
		simulate_fuzzy_strategy(simulators[i], hmm_model, full_returns, hidden_states)
		for i in range(sim_count)
	]

	# Print stats
	avg_return = sum(results) / sim_count
	print(f"Fuzzy strategy: Avg final capital after {sim_count} runs = {avg_return:.4f}")

	if avg_return > 1.0:
		print("Recommendation: Fuzzy-based strategy appears profitable.")
	else:
		print("Recommendation: Fuzzy-based strategy underperforms or is neutral.")
