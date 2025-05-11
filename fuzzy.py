import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# 1. Define fuzzy input and output variables on a scale from 0 to 100
# These variables are used to model expert knowledge in linguistic terms
volatility = ctrl.Antecedent(np.arange(0, 101, 1), 'volatility')  # input: market volatility
market_state = ctrl.Antecedent(np.arange(0, 101, 1), 'market_state')  # input: market trend
investment_attractiveness = ctrl.Consequent(np.arange(0, 101, 1), 'investment_attractiveness')  # output: how attractive is the investment

# 2. Assign membership functions to each fuzzy variable
# Membership functions define what "low", "medium", and "high" mean for each variable
# In ./resources/membership_functions.png, you can see plotted membership functions for each variable.
volatility['low'] = fuzz.trimf(volatility.universe, [0, 0, 50])
volatility['medium'] = fuzz.trimf(volatility.universe, [25, 50, 75])
volatility['high'] = fuzz.trimf(volatility.universe, [50, 100, 100])

market_state['decline'] = fuzz.trimf(market_state.universe, [0, 0, 50])
market_state['stable'] = fuzz.trimf(market_state.universe, [25, 50, 75])
market_state['growth'] = fuzz.trimf(market_state.universe, [50, 100, 100])

investment_attractiveness['low'] = fuzz.trimf(investment_attractiveness.universe, [0, 0, 50])
investment_attractiveness['medium'] = fuzz.trimf(investment_attractiveness.universe, [25, 50, 75])
investment_attractiveness['high'] = fuzz.trimf(investment_attractiveness.universe, [50, 100, 100])

# 3. Define fuzzy logic rules that link inputs to the output
# Each rule follows the pattern: IF condition THEN result
rule1 = ctrl.Rule(market_state['growth'] & volatility['low'], investment_attractiveness['high'])
rule2 = ctrl.Rule(market_state['growth'] & volatility['medium'], investment_attractiveness['medium'])
rule3 = ctrl.Rule(market_state['growth'] & volatility['high'], investment_attractiveness['medium'])

rule4 = ctrl.Rule(market_state['stable'] & volatility['low'], investment_attractiveness['high'])
rule5 = ctrl.Rule(market_state['stable'] & volatility['medium'], investment_attractiveness['medium'])
rule6 = ctrl.Rule(market_state['stable'] & volatility['high'], investment_attractiveness['low'])

rule7 = ctrl.Rule(market_state['decline'] & volatility['low'], investment_attractiveness['medium'])
rule8 = ctrl.Rule(market_state['decline'] & volatility['medium'], investment_attractiveness['low'])
rule9 = ctrl.Rule(market_state['decline'] & volatility['high'], investment_attractiveness['low'])

# 4. Create the fuzzy control system based on all defined rules
# This system can now evaluate the output given any pair of inputs
attractiveness_ctrl = ctrl.ControlSystem([
	rule1, rule2, rule3,
	rule4, rule5, rule6,
	rule7, rule8, rule9
])

# 5. Function to compute output value using the fuzzy system
# This function takes numerical inputs, runs fuzzy logic, and returns a crisp result
def get_fast_attractiveness(simulator, market_state_value, volatility_value):
	simulator.input['market_state'] = market_state_value
	simulator.input['volatility'] = volatility_value
	simulator.compute()
	return simulator.output['investment_attractiveness']
