from markov import MarkovChain

if __name__ == "__main__":
    states = ['Growth', 'Stable', 'Decline']
    transitions = [
        [0.6, 0.3, 0.1], # Growth
        [0.2, 0.6, 0.2], # Stable
        [0.1, 0.4, 0.5] # Decline
    ]

    market_chain = MarkovChain(states, transitions)
    print(market_chain.generate_states('Growth', 10))

