from markov import MarkovChain
from fuzzy import get_fast_attractiveness, attractiveness_ctrl
import numpy as np
import skfuzzy.control as ctrl

if __name__ == "__main__":
	states = ['Growth', 'Stable', 'Decline']
	transitions = [
		[0.6, 0.3, 0.1],    # From Growth
		[0.2, 0.6, 0.2],    # From Stable
		[0.1, 0.4, 0.5]     # From Decline
	]

	# Define returns for each market state
	returns = {
		'Growth': 0.05,
		'Stable': 0.00,
		'Decline': -0.04
	}

	# Map market states to fuzzy numeric scale
	market_state_score = {
		'Decline': 20,
		'Stable': 50,
		'Growth': 80
	}

	market_chain = MarkovChain(states, transitions)

	# Simulation parameters
	sim_count = 1000
	days = 100

	# Pre-create fuzzy simulators to avoid recomputation overhead
	simulators = [ctrl.ControlSystemSimulation(attractiveness_ctrl) for _ in range(sim_count)]

	def simulate_fuzzy_strategy(simulator):
		state = 'Stable'
		total_return = 1.0

		for _ in range(days):
			# Generate synthetic volatility (0 to 100)
			volatility = np.random.uniform(0, 100)
			state_score = market_state_score[state]

			# Evaluate attractiveness
			attractiveness = get_fast_attractiveness(simulator, state_score, volatility)

			if attractiveness > 60:
				r = returns[state]
				total_return *= (1 + r)

			state = market_chain.next_state(state)

		return total_return

	# Run simulations
	fuzzy_results = [
		simulate_fuzzy_strategy(simulators[i])
		for i in range(sim_count)
	]

	avg_fuzzy = sum(fuzzy_results) / sim_count

	print(f"Fuzzy strategy: Avg final capital after {sim_count} runs = {avg_fuzzy:.4f}")

	if avg_fuzzy > 1.0:
		print("Recommendation: Fuzzy-based strategy appears profitable.")
	else:
		print("Recommendation: Fuzzy-based strategy underperforms or is neutral.")
