import plotly.express as px
from plotly.express.colors import sample_colorscale
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from copy import deepcopy
from membrane import Membrane_jax

def scatter_animation(model=None, annotate_molecule=False, speed=50, subset=None, prev=None, table=None, withinit=True, **kwargs):
    if model is not None:
        simulations = model.datacollector.get_agent_vars_dataframe(withinit=withinit)
        if annotate_molecule:
            actin_simulations = model.actin_datacollector.get_agent_vars_dataframe(withinit=withinit).rename(columns={'actin_pos_x':'pos_x', 'actin_pos_y':'pos_y'})
            actin_simulations.loc[:, 'molecule'] = 'actin'
            simulations.loc[:, 'molecule'] = 'lipid'
            actin_simulations.loc[:, 'size'] = float(model.actin_r0)
            simulations.loc[:, 'size'] = float(model.r0)
            simulations = pd.concat([simulations, actin_simulations])
    else:
        simulations = table
        
    if 'Step' not in simulations.columns:
        simulations.reset_index(inplace=True)
    
    # color of markers by type of molecule
    if not annotate_molecule:
        color = None
    else:
        color = 'molecule'

    # add prev
    if prev is not None:
        simulations = pd.concat([prev, simulations], sort=True, ignore_index=True)
    
    # subset of frames
    if subset is not None:
        simulations = simulations.loc[simulations['Step'].isin(subset)]
        
    # marker indicating lipid velocity size & direction
    lipid_subset = simulations.loc[simulations['molecule'] == 'lipid']
    lipid_vels = np.nan_to_num(lipid_subset.loc[: , ['vx', 'vy']].values)
    lipid_angles = 90 - (np.nan_to_num(np.arctan2(lipid_vels[:,1], lipid_vels[:,0])) / np.pi * 180)
    lipid_vsizes = np.nan_to_num(np.linalg.norm(lipid_vels, axis=1))
    # for each frame
    frame_steps = lipid_subset.drop_duplicates('Step').Step.values
    frame_lipid_angles = [lipid_angles[(lipid_subset['Step'] == step).values] for step in frame_steps]
    frame_lipid_vsize = [lipid_vsizes[(lipid_subset['Step'] == step).values] for step in frame_steps]
    max_line_width = 2
    frame_line_widths = [np.nan_to_num((lipid_vsize - lipid_vsize.min()) / (lipid_vsize.max() - lipid_vsize.min())) * max_line_width for lipid_vsize in frame_lipid_vsize]
    simulations = simulations.loc[simulations['Step'].isin(frame_steps)]

    # visualize
    xpos_colname = kwargs.get('xpos_colname', 'pos_x')
    ypos_colname = kwargs.get('ypos_colname', 'pos_y')
    fig = px.scatter(simulations, x=xpos_colname, y=ypos_colname, 
            animation_frame='Step', animation_group='AgentID', 
            hover_name='AgentID', width=950, height=900, 
            size='size', size_max=15, 
            color=color, range_color=[0, 3],
            range_x=[30,110])
    # mark lipid velocity for figure
    symbol = 'triangle-up'
    fig.update_traces(
        selector=dict(name='lipid'),
        marker=dict(symbol=symbol, angle=frame_lipid_angles[0],
                    line=dict(width=frame_line_widths[0], color='DarkSlateGrey')))
    # mark lipid velocity for frames
    for i in range(len(frame_steps)):
        fig.frames[i].data[0].update(marker=dict(symbol=symbol, angle=frame_lipid_angles[i],),
                                    line=dict(color='DarkSlateGrey'))
        fig.frames[i].data[0].marker['line']['width'] = frame_line_widths[i]
    # color new lipids
    agents_iterations = model.agents_iterations if model is not None else kwargs.get('agents_iterations', None)
    if agents_iterations is not None:
        num_interval_colors = kwargs.get('num_interval_colors', 20)
        interval_frame_size = kwargs.get('interval_frame_size', 1)
        fig = color_new_lipids(fig, agents_iterations, 
                               num_interval_colors=num_interval_colors, 
                               interval_frame_size=interval_frame_size)
    
    # speed
    fig.layout.updatemenus[0]['buttons'][0]['args'][1]['frame']['duration'] = speed

    # axis range
    axis_range = kwargs.get('axis_range', 70)
    x_axist_start = kwargs.get('x_axist_start', 0)
    _ = fig.update_xaxes(dict(range=(x_axist_start, x_axist_start+axis_range*2), autorange=False))
    _ = fig.update_yaxes(dict(range=(-axis_range, axis_range), autorange=False))

    return fig, simulations

def annotate_velocity(fig, simulations):
    # num of frames
    num_frames = len(simulations.groupby('Step'))
    # add annotations to fig
    for i in range(1, num_frames):
        status = simulations.loc[simulations['Step'] == i]
        fig.frames[i].layout.annotations = [go.layout.Annotation(x=status['pos_x'].values, 
                                                                y=status['pos_y'].values,
                                                                ax=status['pos_x'].values+status['vx'].values,
                                                                ay=status['pos_y'].values+status['vy'].values, 
                                                                showarrow=True)]
    return fig

def color_new_lipids(fig, agents_iterations, num_interval_colors=10, interval_frame_size=1):
    """_summary_

    Args:
        fig (plotly.graph_objs._figure.Figure): plotly figure
        agents_iterations (np.array(N,)): iterations of each agent/lipid
        num_interval_colors (int, optional): gradual colors for new lipids. Defaults to 10.
        interval_frame_size (int, optional): interval for new lipids to change ot next color. Defaults to 1.
    """
    # frames when adding new lipids
    agents_iterations = np.array(agents_iterations)
    num_steps = agents_iterations.max()
    num_frames = len(fig.frames)
    frame_iterval = num_steps // num_frames
    appear_frames = num_frames - np.array(agents_iterations) // frame_iterval 
    addlipid_frames = np.sort(np.unique(appear_frames[appear_frames > 0]))
    # colors for new lipids
    colors = sample_colorscale(['rgb(103,0,31)','#636efa'], samplepoints=np.linspace(0,1,num_interval_colors))
    colors = [color.replace(' ','') for color in colors]
    # for each frame with new lipids, color new lipids
    for frame in addlipid_frames:
        # new lipids to color
        frame_newlipids_mask = (appear_frames == frame)
        # frames to change color
        color_frames = frame + np.arange(num_interval_colors) * interval_frame_size
        # color new lipids
        for color_frame, color in zip(color_frames, colors):
            if color_frame >= num_frames:
                break
            current_newlipid_mask = frame_newlipids_mask[appear_frames<=color_frame]
            agents_colors = fig.frames[color_frame].data[0].marker['color']
            if isinstance(agents_colors, str):
                agents_colors = np.array([agents_colors] * len(current_newlipid_mask), dtype='<U15')
            elif isinstance(agents_colors, np.ndarray):
                agents_colors = np.copy(agents_colors)
            agents_colors[current_newlipid_mask] = [color] * current_newlipid_mask.sum()
            fig.frames[color_frame].data[0].marker['color'] = agents_colors
    return fig    

def metric_bar_animation(model, speed=50, subset=None, withinit=True):
    metrics = model.datacollector.get_model_vars_dataframe(withinit).drop(columns=['center_x','center_y'])
    # fold changes of metrics
    ratio = (metrics / metrics.loc[0]).stack().reset_index()
    ratio.columns = ['Step', 'Metric', 'Ratio']
    # add log10 of ratios
    ratio.loc[:, 'log10'] = np.log10(ratio['Ratio'])

    # subset of frames
    if subset is not None:
        ratio = ratio.loc[ratio['Step'].isin(subset)]

    # extreme values of ratios
    max_value = np.abs(ratio['log10']).max()

    # visualize
    fig = px.bar(ratio, x='Metric', y='log10', color='Metric', animation_frame='Step', 
            width=900, height=900, range_y=[-max_value*1.5, max_value*1.5])

    # speed
    fig.layout.updatemenus[0]['buttons'][0]['args'][1]['frame']['duration'] = speed

    return fig, ratio

def metric_scatter_animation(model=None, speed=50, subset=None, prev=None, table=None, withinit=True):
    
    if model is not None:
        metrics = model.datacollector.get_model_vars_dataframe(withinit).drop(columns=['center_x','center_y'])
        # add current status
        metrics.loc[len(metrics)*model.jump_step] = [getattr(model, attr) for attr in metrics.columns]
        # fold changes of metrics
        ratio = (metrics / metrics.iloc[0]).stack().reset_index()
        ratio.columns = ['Step', 'Metric', 'Ratio']
        ratio = ratio.applymap(lambda v: np.asarray(v))
        # add log10 of ratios
        ratio.loc[:, 'log10'] = np.log10(ratio['Ratio'])
    else:
        ratio = table

    # add previous status
    if prev is not None:
        last_step = prev.Step.max()
        ratio.loc[:, 'Step'] = ratio['Step'] + last_step
        last_ratio = prev.loc[prev['Step'] == last_step, 'log10'].values
        ratio.loc[:, 'log10'] = (ratio['log10'].values.reshape(-1, len(last_ratio)) + last_ratio).reshape(-1)
        ratio = pd.concat([prev, ratio], sort=True, ignore_index=True)
        ratio.drop_duplicates(['Step', 'Metric'], inplace=True)

    # subset of frames
    if subset is not None:
        ratio = ratio.loc[ratio['Step'].isin(subset)]

    # extreme values of ratios
    max_value = np.abs(ratio['log10']).max()

    # layout
    layout = go.Layout(
        {
            'height': 450,
            'legend': {'title': {'text': 'Metric'}, 'tracegroupgap': 0},
            # 'showlegend': False,
            'margin': {'t': 60},
            'template': 'plotly',
            'updatemenus': [{'buttons': [{'args': [None, {'frame': {'duration': speed,
                                                'redraw': True}, 'mode': 'immediate',
                                                'fromcurrent': True, 'transition':
                                                {'duration': 500, 'easing': 'linear'}}],
                                        'label': '&#9654;',
                                        'method': 'animate'},
                                        {'args': [[None], {'frame': {'duration': 0,
                                                'redraw': True}, 'mode': 'immediate',
                                                'fromcurrent': True, 'transition':
                                                {'duration': 0, 'easing': 'linear'}}],
                                        'label': '&#9724;',
                                        'method': 'animate'}],
                            'direction': 'left',
                            'pad': {'r': 10, 't': 70},
                            'showactive': False,
                            'type': 'buttons',
                            'x': 0.1,
                            'xanchor': 'right',
                            'y': 0,
                            'yanchor': 'top'}],
            'width': 900,
            'xaxis': {'anchor': 'y', 'autorange': False, 
                    'domain': [0.0, 1.0], 
                    'range': [ratio.Step.min()*1.01, ratio.Step.max()*1.01], 
                    'title': {'text': 'Step'}},
            'yaxis': {'anchor': 'x', 'autorange': False,
                    'domain': [0.0, 1.0],
                    'range': [-max_value*1.5, max_value*1.5],
                    'title': {'text': 'log10'}}
    })

    # colors
    palette = px.colors.qualitative.Plotly
    metrics = list(ratio.Metric.unique())
    marker_colors = {metric: palette[i] for i, metric in enumerate(metrics)}

    # traces
    traces = [go.Scatter(x=subset.Step, y=subset.log10, mode='lines', 
            line=dict(width=2, color=marker_colors[metric]), 
            name=metric, showlegend=False) 
            for metric, subset in ratio.groupby('Metric')]
    traces = traces + deepcopy(traces)
    for idx in range(len(traces)//2):
        traces[idx].showlegend = True

    # frames
    frames = [go.Frame(
        data=[go.Scatter(x=msubset.Step, y=msubset.log10, mode='markers', 
            marker=dict(color=marker_colors[metric], size=10), showlegend=True) 
            for metric, msubset in subset.groupby('Metric')]) 
            for step, subset in ratio.groupby('Step')]

    # figures
    fig = go.Figure(data=traces, layout=layout, frames=frames)

    return fig, ratio

# combine metric fig & scatter fig
def combine_metric_scatter(fig, metric_fig, annotate_molecule=False):
    fig = deepcopy(fig)
    # update layout
    updated_layout = fig.update_layout({'xaxis2':{'anchor':'y2', 'domain':[0.,1.], 
                                            'range':metric_fig.layout.xaxis.range, 
                                            'title':metric_fig.layout.xaxis.title},
                                        'yaxis':{'anchor':'x', 'domain':[0.2,1.]},
                                        'yaxis2':{'anchor':'x2', 'domain':[0.,0.15], 
                                            'range':metric_fig.layout.yaxis.range, 
                                            'title':metric_fig.layout.yaxis.title},
                                        'height': 1100,
                                        'legend_tracegroupgap':10 if annotate_molecule else 180,})

    # update traces
    _ = fig.add_traces(metric_fig.data)

    # change axis of newly added traces
    begin = 1 if not annotate_molecule else 2
    for idx in range(begin, len(fig.data)):
        fig.data[idx].xaxis = 'x2'
        fig.data[idx].yaxis = 'y2'

    # update frames
    for idx in range(len(metric_fig.frames)):
        orig_data = [fig.frames[idx].data[0]]
        if annotate_molecule:
            orig_data = orig_data + [fig.frames[idx].data[1]]
        metric_data = list(metric_fig.frames[idx].data)
        for midx in range(len(metric_data)):
            metric_data[midx].xaxis = 'x2'
            metric_data[midx].yaxis = 'y2'
        fig.frames[idx].update(data=orig_data + metric_data)
    
    if not annotate_molecule:
        # change positions of the color axis
        _ = fig.update_coloraxes({'colorbar':{'len':0.6, 'y':0.65}})
        # change positions of legend
        _ = fig.update_layout({'legend':{'y':0.1}})

    return fig