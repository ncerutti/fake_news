# Fake News and Investment ABM

This project implements an Agent-Based Model (ABM) to explore the impact of misinformation, particularly concerning climate change, on investment decisions and market dynamics.

## Core Idea

The model simulates a market with two types of investors:

1.  **Individual Investors:** Make investment decisions based on personal beliefs (e.g., climate change risk perception). These beliefs can be influenced by exposure to misinformation spreading through a network.
2.  **Institutional Investors:** Base their decisions more on aggregate market sentiment and trends, potentially amplifying or mitigating the effects of individual biases.

Agents allocate capital between asset classes (e.g., 'Green' vs. 'Fossil'). The model investigates how the spread of misinformation can shift beliefs, alter investment patterns, and potentially lead to market distortions like asset mispricing or increased volatility.

## Implementation

The model is built using the [Mesa](https://github.com/projectmesa/mesa) framework in Python.

-   **`model.py`**: Defines the main `FakeNewsInvestmentModel` class, simulation steps, and data collection.
-   **`agents.py`**: Defines the `IndividualInvestor` and `InstitutionalInvestor` agent classes.
-   **`server.py`**: (Optional) Provides a browser-based visualization using Mesa's server component.
-   **`requirements.txt`**: Lists project dependencies.

## Running the Model

(Instructions to be added once the basic implementation is complete)