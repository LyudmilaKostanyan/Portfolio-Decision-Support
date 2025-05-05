import numpy as np

class MarkovChain:
	#states = ['Growth', 'Stable', 'Decline']
	#transition_matrix: [i][j] is the probability of transition from state i to state j
	def __init__(self, states, transition_matrix):
		self.states = states
		self.transition_matrix = np.array(transition_matrix)
		self.index_dict = {state: index for index, state in enumerate(states)}

	# Returns the next state based on transition probabilities from the current state
	def next_state(self, current_state):
		state_index = self.index_dict[current_state]
		# Controlled randomness: next state is sampled according to predefined probabilities
		next_state = np.random.choice(
			self.states,
			p=self.transition_matrix[state_index]
		)
		return next_state

	# Generate a sequence of future states
	def generate_states(self, current_state, num=10):
		future_states = [current_state]
		for _ in range(num - 1):
			next_s = self.next_state(future_states[-1])
			future_states.append(next_s)
		return future_states

