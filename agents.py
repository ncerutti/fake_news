import mesa


class InvestorAgent(mesa.Agent):
    """Base class for all investor agents."""
    def __init__(self, model, initial_belief, initial_portfolio):
        super().__init__(model)
        # Belief about climate change risk (0: Denier, 1: Strong Believer)
        self.belief_climate_risk = initial_belief
        # Portfolio allocation {'Green': fraction, 'Fossil': fraction}
        self.portfolio = initial_portfolio
        self.wealth = 1.0  # Start with normalized wealth

    def step(self):
        """Agent's action during a simulation step."""
        self.update_beliefs()
        self.allocate_portfolio()
        # Placeholder for wealth changes based on asset performance
        self.update_wealth()

    def update_beliefs(self):
        """Placeholder for belief update logic (e.g., based on news)."""
        pass  # To be implemented

    def allocate_portfolio(self):
        """Placeholder for portfolio allocation logic."""
        pass  # To be implemented

    def update_wealth(self):
        """Placeholder for wealth update based on portfolio performance."""
        # This would typically depend on market price changes defined
        # in the model step
        pass  # To be implemented


class IndividualInvestor(InvestorAgent):
    """Represents an individual investor whose decisions are belief-driven."""
    def __init__(self, model, initial_belief, susceptibility_to_misinfo):
        # Initial portfolio allocation based on belief
        # Simple example: more belief -> more green
        initial_portfolio = {
            'Green': initial_belief,
            'Fossil': 1.0 - initial_belief
        }
        super().__init__(model, initial_belief, initial_portfolio)
        # How easily swayed by fake news (0 to 1)
        self.susceptibility = susceptibility_to_misinfo

    def update_beliefs(self):
        """
        Update belief based on exposure to news (real or fake) from the model.
        """
        # Example: Check for news in the model environment
        news_type, news_strength = \
            self.model.get_news_for_agent(id(self))

        if news_type == "fake_denial":
            # Decrease belief, influenced by susceptibility
            # More change if belief is already low
            change = (news_strength * self.susceptibility *
                      (1 - self.belief_climate_risk))
            self.belief_climate_risk = max(
                0, self.belief_climate_risk - change)
        elif news_type == "real_climate":
            # Increase belief, perhaps less influenced by susceptibility?
            # More change if belief is already high
            change = (news_strength * (1 - self.susceptibility) *
                      self.belief_climate_risk)
            self.belief_climate_risk = min(
                1, self.belief_climate_risk + change)
        # Add more news types? Neutral news?

    def allocate_portfolio(self):
        """Allocate portfolio directly based on current belief."""
        # Simple linear allocation based on belief
        self.portfolio['Green'] = self.belief_climate_risk
        self.portfolio['Fossil'] = 1.0 - self.belief_climate_risk
        # Ensure normalization (might be slightly off due to float precision)
        total = sum(self.portfolio.values())
        if total > 0:
            self.portfolio['Green'] /= total
            self.portfolio['Fossil'] /= total


class InstitutionalInvestor(InvestorAgent):
    """Represents an institutional investor influenced by market sentiment."""
    def __init__(self, model, sentiment_sensitivity):
        # Institutions might start neutral or follow initial market average
        initial_belief = 0.5  # Start neutral
        initial_portfolio = {'Green': 0.5, 'Fossil': 0.5}
        super().__init__(model, initial_belief, initial_portfolio)
        # How much they follow the crowd (0 to 1)
        self.sentiment_sensitivity = sentiment_sensitivity

    def update_beliefs(self):
        """
        Institutional 'belief' is more like a reflection of market sentiment.
        """
        # This agent doesn't update personal belief based on news
        # in the same way. Instead, its allocation logic uses market
        # sentiment directly. We can still use the 'belief' field to
        # sentiment.
        self.belief_climate_risk = self.model.get_market_sentiment()

    def allocate_portfolio(self):
        """Allocate portfolio based on market sentiment and sensitivity."""
        # e.g., average belief of individuals
        market_sentiment = self.model.get_market_sentiment()

        # Blend own 'neutral' view with market sentiment based on sensitivity
        target_green_alloc = ((1 - self.sentiment_sensitivity) * 0.5 +
                              self.sentiment_sensitivity * market_sentiment)

        self.portfolio['Green'] = target_green_alloc
        self.portfolio['Fossil'] = 1.0 - target_green_alloc
        # Ensure normalization
        total = sum(self.portfolio.values())
        if total > 0:
            self.portfolio['Green'] /= total
            self.portfolio['Fossil'] /= total