# Hybrid Investment Decision Support System

This project implements a hybrid **Decision Support System (DSS)** that combines **Markov Chains** for long-term forecasting and **Fuzzy Inference Systems (FIS)** for short-term trading decisions. It is built to analyze market data, simulate two different investment strategies, and compare them to actual market outcomes using real stock price histories.

## Project Overview

Traditional investment decision models often rely purely on statistical trends or machine learning forecasts. In contrast, this project integrates:

* **Markov Chains**: for modeling the long-term transition of market conditions.
* **Fuzzy Logic**: for expert-based reasoning about daily trading actions, based on market volatility and trend strength.

By combining these approaches, the system adapts to both structural patterns and short-term market behavior.

## Features

* Fetches historical stock data using Yahoo Finance
* Classifies market states as Growth, Stable, or Decline
* Constructs a state transition matrix for Markov modeling
* Runs Monte Carlo simulations for long-hold Markov strategies
* Applies fuzzy logic rules for day-by-day entry/exit decisions
* Compares results from three approaches:

  * Actual historical return
  * FIS-based dynamic trading strategy
  * Markov-hold simulation strategy

## File Structure

```
.
├── main.py           # Core runner: downloads data, runs strategies, prints summary
├── fuzzy.py          # Fuzzy Inference System (FIS) definition and rule base
├── markov.py         # MarkovChain class for modeling state transitions
```

## Strategy Descriptions

### Markov-Hold Strategy (Long-Term)

* Analyzes historical returns and classifies each day
* Builds a transition probability matrix between states
* Simulates a one-year trajectory of state transitions using Monte Carlo sampling
* Applies the average daily return of each state to simulate capital evolution
* Returns the mean performance over multiple runs

### Fuzzy Logic Strategy (Short-Term)

* For each trading day, calculates:

  * Market Score: deviation from 30-day moving average
  * Volatility Score: standard deviation of recent returns
* Uses fuzzy rules like:

  * If market is growing and volatility is low, attractiveness is high
* Based on output:

  * If attractiveness > 60: Buy
  * If attractiveness < 40: Sell
  * Otherwise: Hold

## Example Output

When running the system on the MULN ticker in backtest mode:

```
If you had invested $1,000 in MULN one year ago:
	• Today it would be worth $0.00,
	  a profit of $-1,000.00 (-100.00%).

Under the FIS-based trading strategy:
	• Your capital would be $1,000.00,
	  a profit of $0.00 (0.00%).

Under the Markov-hold strategy:
	• Your capital would be $0.00,
	  a profit of $-1,000.00 (-100.00%).
```

## How to Use

### Installation

```bash
pip install numpy scipy packaging networkx scikit-fuzzy yfinance pandas matplotlib
```

### Run Backtest Mode

Simulates actual past year using real historical prices:

```bash
python main.py --mode backtest --ticker MULN
```

### Run Forecast Mode

Simulates expected performance for the next year based on current state:

```bash
python main.py --mode forecast --ticker MULN
```

## Fuzzy Logic Rule Base

| Market State | Volatility | Attractiveness |
| ------------ | ---------- | -------------- |
| Growth       | Low        | High           |
| Growth       | Medium     | Medium         |
| Growth       | High       | Medium         |
| Stable       | Low        | High           |
| Stable       | Medium     | Medium         |
| Stable       | High       | Low            |
| Decline      | Low        | Medium         |
| Decline      | Medium     | Low            |
| Decline      | High       | Low            |

## Purpose and Context

This project was developed for a diploma thesis on:

**Optimization of Decision Support Systems Using Markov Chains and Fuzzy Logic**

Its objective is to explore hybrid modeling for investment strategy selection and validate it through comparison against real-world financial performance.
