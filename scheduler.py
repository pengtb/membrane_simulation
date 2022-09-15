from mesa.time import BaseScheduler

class SimultaneousActivation(BaseScheduler):
    """A scheduler to simulate the simultaneous activation of all the agents.

    This scheduler requires that each agent have two methods: step and advance.
    step() activates the agent and stages any necessary changes, but does not
    apply them yet. advance() then applies the changes.

    """

    def step(self, **kwargs) -> None:
        """Step all agents, then advance them."""
        agent_keys = list(self._agents.keys())
        for agent_key in agent_keys:
            self._agents[agent_key].step(**kwargs)
        for agent_key in agent_keys:
            self._agents[agent_key].advance(**kwargs)
        self.steps += 1
        self.time += 1

class SimpleSimultaneousActivation(BaseScheduler):
    """A scheduler to simulate the simultaneous activation of all the agents.

    This scheduler requires that each agent have two methods: step and advance.
    step() activates the agent and stages any necessary changes, but does not
    apply them yet. advance() then applies the changes.

    """

    def step(self, **kwargs) -> None:
        """Step all agents, then advance them."""
        self.steps += 1
        self.time += 1