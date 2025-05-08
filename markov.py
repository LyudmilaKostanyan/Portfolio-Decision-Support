import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt

class HiddenMarkovModel:
	def __init__(self, n_states=3):
		#Initialize a Gaussian HMM with specified number of hidden states.
		self.model = hmm.GaussianHMM(
			n_components=n_states,
			covariance_type="diag",
			n_iter=100
		)
		self.n_states = n_states

		# Optional: human-readable names for states
		self.hidden_states_names = ['Bullish', 'Neutral', 'Bearish']

	def train(self, observations):
		"""
		Train the HMM using the given observation sequence.
		:param observations: numpy array of shape (T, 1), e.g. daily returns
		"""
		self.model.fit(observations)

	def predict_states(self, observations):
		"""
		Predict the most likely hidden state sequence given observations.
		:param observations: same shape as used in training
		:return: list of state indices
		"""
		return self.model.predict(observations)

	def print_states_sequence(self, hidden_states):
		"""
		Print the sequence of state names corresponding to state indices.
		:param hidden_states: list or array of integers
		"""
		named_states = [self.hidden_states_names[state] for state in hidden_states]
		print("Inferred hidden market states sequence:")
		print(named_states)

	def plot_hidden_states(self, observations, hidden_states):
		"""
		Plot the observations and overlay background colors based on hidden states.
		:param observations: array of returns
		:param hidden_states: sequence of predicted state indices
		"""
		plt.figure(figsize=(12, 6))
		plt.plot(observations, label='Returns', linewidth=1.5)
		plt.title('Hidden Market States (HMM)')
		plt.xlabel('Time')
		plt.ylabel('Return')
		colors = ['green', 'orange', 'red']
		labels_seen = set()

		for idx, state in enumerate(hidden_states):
			label = self.hidden_states_names[state]
			color = colors[state]
			if label not in labels_seen:
				plt.axvspan(idx - 0.5, idx + 0.5, color=color, alpha=0.3, label=label)
				labels_seen.add(label)
			else:
				plt.axvspan(idx - 0.5, idx + 0.5, color=color, alpha=0.3)

		plt.legend()
		plt.grid(True)
		plt.tight_layout()
		plt.show()
