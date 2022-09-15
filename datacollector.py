from mesa.datacollection import DataCollector
import types
from functools import partial
import pandas as pd
import numpy as np

class JumpDataCollector(DataCollector):
    def __init__(self, **kwargs):
        model_reporters = kwargs.get('model_reporters', None)
        agent_reporters = kwargs.get('agent_reporters', None)
        tables = kwargs.get('tables', None)
        super().__init__(model_reporters=model_reporters, agent_reporters=agent_reporters, tables=tables)
        self.jump_step = kwargs.get('jump_step', 1)

    def collect(self, model):
        """Collect all the data for the given model object."""
        if model.schedule.steps % self.jump_step == 0:
            if self.model_reporters:
                for var, reporter in self.model_reporters.items():
                    # Check if Lambda operator
                    if isinstance(reporter, types.LambdaType):
                        self.model_vars[var].append(reporter(model))
                    # Check if model attribute
                    elif isinstance(reporter, partial):
                        self.model_vars[var].append(reporter(model))
                    # Check if function with arguments
                    elif isinstance(reporter, list):
                        self.model_vars[var].append(reporter[0](*reporter[1]))
                    else:
                        self.model_vars[var].append(self._reporter_decorator(reporter))

            if self.agent_reporters:
                agent_records = self._record_agents(model)
                self._agent_records[model.schedule.steps] = list(agent_records)
                
    def get_model_vars_dataframe(self):
        """Create a pandas DataFrame from the model variables.

        The DataFrame has one column for each model variable, and the index is
        (implicitly) the model tick.

        """
        model_vars_table = pd.DataFrame(self.model_vars)
        model_vars_table.index = model_vars_table.index * self.jump_step
        return model_vars_table

class JumpModelDataCollector(DataCollector):
    def __init__(self, model, **kwargs):
        model_reporters = kwargs.get('model_reporters', None)
        agent_reporters = kwargs.get('agent_reporters', None)
        tables = kwargs.get('tables', None)
        super().__init__(model_reporters=model_reporters, agent_reporters=agent_reporters, tables=tables)
        self.jump_step = kwargs.get('jump_step', 1)
        self.all_agents_records = []
        self.model = model

    def collect(self, model):
        """Collect all the data for the given model object."""
        if model.schedule.steps % self.jump_step == 0:
            if self.model_reporters:
                for var, reporter in self.model_reporters.items():
                    # Check if Lambda operator
                    if isinstance(reporter, types.LambdaType):
                        self.model_vars[var].append(reporter(model))
                    # Check if model attribute
                    elif isinstance(reporter, partial):
                        self.model_vars[var].append(reporter(model))
                    # Check if function with arguments
                    elif isinstance(reporter, list):
                        self.model_vars[var].append(reporter[0](*reporter[1]))
                    else:
                        self.model_vars[var].append(self._reporter_decorator(reporter))

            if self.agent_reporters:
                agent_records = []
                for var, reporter in self.agent_reporters.items():
                    # Check if Lambda operator
                    if isinstance(reporter, types.LambdaType):
                        agent_records.append(reporter(model))
                    # Check if model attribute
                    elif isinstance(reporter, partial):
                        agent_records.append(reporter(model))
                    # Check if function with arguments
                    elif isinstance(reporter, list):
                        agent_records.append(reporter[0](*reporter[1]))
                    else:
                       agent_records.append(self._reporter_decorator(reporter))
                self.all_agents_records.append(np.stack(agent_records, axis=-1))
                
    def get_model_vars_dataframe(self):
        """Create a pandas DataFrame from the model variables.

        The DataFrame has one column for each model variable, and the index is
        (implicitly) the model tick.

        """
        model_vars_table = pd.DataFrame(self.model_vars)
        model_vars_table.index = model_vars_table.index * self.jump_step
        return model_vars_table

    def get_agent_vars_dataframe(self):
        """Create a pandas DataFrame from the agent variables.

        The DataFrame has one column for each variable, with two additional
        columns for tick and agent_id.

        """
        # steps
        steps = np.arange(self.jump_step, self.model.schedule.steps+self.jump_step, self.jump_step)
        steps = np.repeat(steps, self.model.N)
        # agent ids
        agent_ids = self.model.agent_ids
        agent_ids = np.tile(agent_ids, len(self.all_agents_records))
        # records 
        records = np.concatenate(self.all_agents_records, axis=0)
        # table
        agent_vars_table = pd.DataFrame(records, columns=self.agent_reporters.keys())
        agent_vars_table.loc[:, 'AgentID'] = agent_ids
        agent_vars_table.loc[:, 'Step'] = steps
        agent_vars_table.set_index(['Step', 'AgentID'], inplace=True)
        return agent_vars_table