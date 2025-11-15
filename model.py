import mesa
from mesa.agent import AgentSet
import random
import numpy as np

from agents import IndividualInvestor, InstitutionalInvestor


class FakeNewsInvestmentModel(mesa.Model):
    """
    An agent-based model simulating the impact of fake news on investment.
    """

    def __init__(
        self,
        n_individual,
        n_institutional,
        initial_belief_distribution="uniform",
        misinfo_susceptibility_distribution="uniform",
        sentiment_sensitivity_distribution="uniform",
        network_structure="random",  # Future: "scale_free", "small_world"
        news_injection_rate=0.1,
        fake_news_proportion=0.5,
        news_strength=0.1,
    ):
        """
        Initialize the model.

        Args:
            n_individual (int): Number of individual investors.
            n_institutional (int): Number of institutional investors.
            initial_belief_distribution (str/tuple): How initial beliefs are set.
                "uniform": random.uniform(0, 1)
                ("normal", mean, std): np.random.normal(mean, std) clipped to [0,1]
            misinfo_susceptibility_distribution (str/tuple): How susceptibility is set.
            sentiment_sensitivity_distribution (str/tuple): How sensitivity is set.
            network_structure (str): Type of social network for news spread.
            news_injection_rate (float): Probability of news appearing each step.
            fake_news_proportion (float): Proportion of injected news that is fake denial.
            news_strength (float): Impact factor of news on belief.
        """
        super().__init__()
        self.num_individual_agents = n_individual
        self.num_institutional_agents = n_institutional
        self.total_agents = n_individual + n_institutional

        # Store parameters
        self.initial_belief_dist = initial_belief_distribution
        self.misinfo_sus_dist = misinfo_susceptibility_distribution
        self.sentiment_sens_dist = sentiment_sensitivity_distribution
        self.network_structure = network_structure  # Placeholder for future use
        self.news_injection_rate = news_injection_rate
        self.fake_news_proportion = fake_news_proportion
        self.current_news = {}  # News present in this step {agent_id: (type, strength)}

        # Create agents (auto-registered with model in Mesa 3.x)
        for i in range(self.num_individual_agents):
            belief = self._get_initial_value(self.initial_belief_dist)
            susceptibility = self._get_initial_value(self.misinfo_sus_dist)
            IndividualInvestor(self, belief, susceptibility)

        for i in range(self.num_institutional_agents):
            sensitivity = self._get_initial_value(self.sentiment_sens_dist)
            # Institutional agents start with neutral belief/portfolio
            InstitutionalInvestor(self, sensitivity)

        # Data collection
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "AvgBelief_Individual": lambda m: np.mean(
                    [
                        a.belief_climate_risk
                        for a in m.agents
                        if isinstance(a, IndividualInvestor)
                    ]
                ),
                "AvgBelief_Institutional": lambda m: np.mean(
                    [
                        a.belief_climate_risk
                        for a in m.agents
                        if isinstance(a, InstitutionalInvestor)
                    ]
                ),
                "AvgPortfolioGreen_Individual": lambda m: np.mean(
                    [
                        a.portfolio.get("Green", 0)
                        for a in m.agents
                        if isinstance(a, IndividualInvestor)
                    ]
                ),
                "AvgPortfolioFossil_Individual": lambda m: np.mean(
                    [
                        a.portfolio.get("Fossil", 0)
                        for a in m.agents
                        if isinstance(a, IndividualInvestor)
                    ]
                ),
                "AvgPortfolioGreen_Institutional": lambda m: np.mean(
                    [
                        a.portfolio.get("Green", 0)
                        for a in m.agents
                        if isinstance(a, InstitutionalInvestor)
                    ]
                ),
                "AvgPortfolioFossil_Institutional": lambda m: np.mean(
                    [
                        a.portfolio.get("Fossil", 0)
                        for a in m.agents
                        if isinstance(a, InstitutionalInvestor)
                    ]
                ),
                "MarketSentiment": lambda m: m.get_market_sentiment(),
            },
            agent_reporters={
                "Belief": "belief_climate_risk",
                "PortfolioGreen": lambda a: a.portfolio.get("Green", 0),
                "PortfolioFossil": lambda a: a.portfolio.get("Fossil", 0),
                "Wealth": "wealth",
            },
        )

        self.running = True
        self.datacollector.collect(self)  # Collect initial state

    def _get_initial_value(self, distribution_config):
        """Helper to get initial value based on distribution config."""
        if isinstance(distribution_config, str) and distribution_config == "uniform":
            return random.uniform(0, 1)
        elif (
            isinstance(distribution_config, tuple)
            and distribution_config[0] == "normal"
        ):
            mean, std = distribution_config[1], distribution_config[2]
            val = np.random.normal(mean, std)
            return np.clip(val, 0, 1)  # Ensure value is within [0, 1]
        else:
            # Default or error
            return random.uniform(0, 1)

    def _inject_news(self):
        """Decide if news is injected and what type/strength."""
        self.current_news = {}  # Clear previous news
        if random.random() < self.news_injection_rate:
            # News is injected this step
            news_strength = random.uniform(0.05, 0.15)  # Example strength range
            if random.random() < self.fake_news_proportion:
                news_type = "fake_denial"
            else:
                news_type = "real_climate"

            # Simplistic: broadcast to all individuals for now
            # Future: Use network structure
            for agent in self.agents:
                if isinstance(agent, IndividualInvestor):
                    self.current_news[id(agent)] = (news_type, news_strength)

    def get_news_for_agent(self, agent_id):
        """Provide news relevant to the agent for this step."""
        return self.current_news.get(agent_id, (None, 0))  # Return (None, 0) if no news

    def get_market_sentiment(self):
        """Calculate the overall market sentiment (e.g., average individual belief)."""
        individual_beliefs = [
            a.belief_climate_risk
            for a in self.agents
            if isinstance(a, IndividualInvestor)
        ]
        if not individual_beliefs:
            return 0.5  # Default if no individuals
        return np.mean(individual_beliefs)

    def step(self):
        """Advance the model by one step."""
        self._inject_news()  # Determine news for this step
        self.agents.shuffle_do("step")  # Agents update beliefs, allocate portfolios
        # Future: Add market price updates based on aggregate demand/supply
        # Future: Add agent wealth updates based on asset performance
        self.datacollector.collect(self)  # Collect data

    def run_model(self, n_steps=100):
        """Run the model for n steps."""
        for i in range(n_steps):
            self.step()


if __name__ == "__main__":
    import os
    from datetime import datetime

    print("--- Script execution started ---")
    N_INDIVIDUAL = 10
    N_INSTITUTIONAL = 50
    N_STEPS = 100
    NEWS_INJECTION_RATE = 0.75
    FAKE_NEWS_PROPORTION = 0.25

    model = FakeNewsInvestmentModel(
        n_individual=N_INDIVIDUAL,
        n_institutional=N_INSTITUTIONAL,
        news_injection_rate=NEWS_INJECTION_RATE,
        fake_news_proportion=FAKE_NEWS_PROPORTION,
    )

    print(
        f"Running model with {N_INDIVIDUAL} individual and "
        f"{N_INSTITUTIONAL} institutional agents for {N_STEPS} steps..."
    )
    print(f"News injection rate: {NEWS_INJECTION_RATE*100}%")
    print(f"Fake news proportion: {FAKE_NEWS_PROPORTION*100}%")

    model.run_model(n_steps=N_STEPS)

    print("\nModel run complete. Accessing collected data...")

    # Get model-level data
    model_data = model.datacollector.get_model_vars_dataframe()
    print("\nModel-level data (first 5 steps):")
    print(model_data.head())
    print("\nModel-level data (last 5 steps):")
    print(model_data.tail())

    # Get agent-level data
    agent_data = model.datacollector.get_agent_vars_dataframe()
    print(f"\nAgent-level data shape: {agent_data.shape}")

    # Create output directory
    output_dir = "simulation_results"
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save data to CSV files
    model_csv = f"{output_dir}/model_data_{timestamp}.csv"
    agent_csv = f"{output_dir}/agent_data_{timestamp}.csv"

    model_data.to_csv(model_csv)
    agent_data.to_csv(agent_csv)

    print(f"\n✓ Data saved successfully!")
    print(f"  - Model data: {model_csv}")
    print(f"  - Agent data: {agent_csv}")

    print("\nNext steps you can try:")
    print("- Adjust news_injection_rate and fake_news_proportion parameters")
    print("- Visualize the data using matplotlib or seaborn")
    print("- Run multiple simulations to compare outcomes")
    print("- Implement network structures for news diffusion")
