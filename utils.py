import re
import json
import plotly

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