import plotly.plotly as py
import plotly.graph_objs as go
import h5py
import os, sys
import numpy as np
import yaml

import edge.util.load_model as load

# if len(sys.argv) <= 2:
#     raise Exception('Usage: python analyze_runs.py NAME RUN_DIR1...')

analysis_name = sys.argv[1]
run_paths = sys.argv[2:]

colors = ['rgba(256,67,67,1)', 'rgba(256,156,67,1)', 'rgba(67,67,256,1)',
    'rgba(67,256,67,1)', 'rgba(256,67,256,1)']


run_paths = []
for rnn_type in ['SRNN', 'EDSRNN', 'GRU', 'EDGRU']:

    for n_unit in [300]:
        match_params = {}
        match_params['rnn_type'] = rnn_type
        match_params['n_unit'] = n_unit

        #
        # print match_params

        run_paths += load.find_matching_model_dir(match_params)


traces = []
hues = np.linspace(0.0, 360.0, len(run_paths) + 1)
for i, run_path in enumerate(run_paths):
    color = 'hsl(' + str(hues[i]) + ',70%,50%)'
    print color

    print run_path

    stats_file = os.path.join(run_path, 'stats.hdf5')
    param_file = os.path.join(run_path, 'params.yaml')

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
            line={'color':color}
        )
        traces.append(train)


        valid_error_hist = f['valid_error_hist'].value
        validate = go.Scatter(
            x = valid_error_hist[0],
            y = valid_error_hist[1],
            name = hparams['run_name']+ '(' +str(hparams['n_unit']) +  ') validation error',
            line = {'color': color, 'dash':'dash'}
        )
        traces.append(validate)



data = go.Data(traces)

layout = go.Layout(
    xaxis= {'title': 'training iteration'},
    yaxis = dict(
        title = 'nats/char',
        # range=[1.0, 2.0]
    )
)
fig = go.Figure(data= data, layout=layout)
unique_url = py.plot(fig, filename = analysis_name)
