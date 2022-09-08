import pandas as pd
import numpy as np

def LoadStatus(status, model):
    # last step
    last_step = status.Step.max()
    # set step to last step
    model.schedule.steps = last_step

    # status of last step
    pos_x = status.loc[status.Step == last_step, 'pos_x'].values
    pos_y = status.loc[status.Step == last_step, 'pos_y'].values
    vx = status.loc[status.Step == last_step, 'vx'].values
    vy = status.loc[status.Step == last_step, 'vy'].values
    # set status of last step
    num_lipids = model.N
    for idx in range(num_lipids):
        model.schedule.agents[idx].pos = np.array((pos_x[idx], pos_y[idx]))
        model.schedule.agents[idx].velocity = np.array((vx[idx], vy[idx]))
    
    return model
