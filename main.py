from markov import MarkovChain

if __name__ == "__main__":
	states = ['Growth', 'Stable', 'Decline']
	transitions = [
		[0.6, 0.3, 0.1],	# Growth
		[0.2, 0.6, 0.2],	# Stable
		[0.1, 0.4, 0.5]		# Decline
	]

	# Define returns for each market state
	returns = {
		'Growth': 0.05,		# +5% return
		'Stable': 0.00,		# 0% return
		'Decline': -0.04	# -4% return
	}

	market_chain = MarkovChain(states, transitions)

	# Simulate a strategy over a given number of days
	def simulate_strategy(strategy_type, days=100):
		# state - starting state
		state = 'Stable'
		# total_return - in start 100% of capital
		total_return = 1.0
		# decline_streak - number of consecutive declines (for strategy B)
		decline_streak = 0

		for _ in range(days):
			if strategy_type == 'B' and decline_streak >= 2:
				# Strategy B: stop investing after 2 consecutive declines
				continue

			if strategy_type == 'A' or decline_streak < 2:
				r = returns[state]
				total_return *= (1 + r)

			# Transition to the next state
			next_s = market_chain.next_state(state)

			# Update decline streak
			if next_s == 'Decline':
				decline_streak += 1
			else:
				decline_streak = 0

			state = next_s

		return total_return

	# Simulate both strategies
	sim_count = 1000
	a_results = []
	b_results = []

	for _ in range(sim_count):
		a_results.append(simulate_strategy('A'))
		b_results.append(simulate_strategy('B'))

	# Calculate average final capital
	avg_a = sum(a_results) / sim_count
	avg_b = sum(b_results) / sim_count

	print(f"Strategy A (Hold): Avg final capital = {avg_a:.4f}")
	print(f"Strategy B (Exit after 2 declines): Avg final capital = {avg_b:.4f}")

	# Compare and print recommendation
	if avg_a > avg_b:
		print("Recommendation: Strategy A (Hold) performs better on average.")
	elif avg_b > avg_a:
		print("Recommendation: Strategy B (Exit after 2 declines) performs better on average.")
	else:
		print("Recommendation: Both strategies perform equally on average.")

