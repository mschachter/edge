import plotly.plotly as py
import plotly.graph_objs as go
import h5py
import os, sys
import numpy as np
import yaml

if len(sys.argv) <= 2:
    raise Exception('Usage: python analyze_runs.py NAME RUN_DIR1...')

analysis_name = sys.argv[1]
run_paths = sys.argv[2:]

colors = ['rgba(256,67,67,1)', 'rgba(256,156,67,1)', 'rgba(67,67,256,1)', 'rgba(67,256,67,1)']

traces = []
for i, run_path in enumerate(run_paths):
    stats_file = os.path.join(run_path, 'stats')
    param_file = os.path.join(run_path, 'params')

    with open(param_file, 'r') as f:
        hparams = yaml.load(f)

    with h5py.File(stats_file) as f:
        train_error_hist = f['train_error_hist'].value
        # import ipdb
        # ipdb.set_trace()
        train = go.Scatter(
            x = train_error_hist[0],
            y = train_error_hist[1],
            name = hparams['run_name'] + '(' +str(hparams['n_unit']) +  ') training error',
            line={'color':colors[i]}
        )
        traces.append(train)

        valid_error_hist = f['valid_error_hist'].value
        validate = go.Scatter(
            x = valid_error_hist[0],
            y = valid_error_hist[1],
            name = hparams['run_name']+ '(' +str(hparams['n_unit']) +  ') validation error',
            line = {'color': colors[i], 'dash':'dash'}
        )
        traces.append(validate)



data = go.Data(traces)

layout = go.Layout(
    xaxis= {'title': 'training iteration'},
    yaxis = dict(
        title = 'nats/char',
        range=[1.0, 2.0]
    )
)
fig = go.Figure(data= data, layout=layout)
unique_url = py.plot(fig, filename = analysis_name)
