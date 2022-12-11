import re
import json
import plotly
import numpy as np
import os

def read_from_html(file):
    """
    load plotly fig from html file
    """
    with open(file, 'r') as f:
        html = f.read()
    call_arg_str = re.findall(r'Plotly\.newPlot\((.*)\)', html)[1]
    call_args = json.loads(f'[{call_arg_str}]')
    plotly_json = {'data': call_args[1], 'layout': call_args[2]}    
    return plotly.io.from_json(json.dumps(plotly_json))

def load_radius(N=1000, all=True, num_neighbor=1, string=True, k=None, eps=None, radius_dir='./radius'):
    """
    load radius from file
    """
    radius_file = f'circle_N{N}'
    if all:
        if string:
            if eps is not None:
                eps = '{:.0e}'.format(eps)
                # eps = eps.replace('0','')
                radius_file += f'_vdwall{eps}'
            if k is not None:
                k = '{:.0e}'.format(k)
                # k = k.replace('0','')
                radius_file += f'_string{k}'
        else:
            radius_file += '_all'
    else:
        radius_file += f'_neighbor{num_neighbor}'
    
    radius_file += '.npy'
    radius_file_path = os.path.join(radius_dir, radius_file)
    if os.path.exists(radius_file_path):
        return np.load(radius_file_path)
    else:
        print('radius file not found')
        return 0.375 / 2
