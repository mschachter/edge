import tensorflow as tf
import yaml
import h5py
import numpy as np
import sys, os

import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools

from edge.networks import Basic_Network

run_path = sys.argv[1]

with open(os.path.join(run_path, 'params.yaml'), 'r') as f:
    hparams = yaml.load(f)


graph = tf.Graph()
with graph.as_default():
    net = Basic_Network(hparams['n_alphabet'], hparams)
    saver = tf.train.Saver()

with tf.Session(graph=graph) as session:
    saver.restore(session, os.path.join(run_path, 'model'))

    R = net.rnn_layer.R.eval(session)
    E = net.rnn_layer.E.eval(session)


R_heat = go.Heatmap(z = R)
E_heat = go.Heatmap(z = E)


fig = tools.make_subplots(rows=1, cols=2, print_grid=False)

fig.append_trace(R_heat, 1,1)
fig.append_trace(E_heat, 1,2)
fig['layout'].update(height=600, width=600, title='Weight matrices')

#py.plot(fig, filename= 'weight analysis')

R_min = np.min(R.flat)
R_max = np.max(R.flat)

n_intervals = 10
R_vals = []
cond_E_mean = []
last_r = None
for r in np.linspace(R_min, R_max, n_intervals + 1):
    if last_r is None:
        last_r = r
        continue
    R_vals.append((r+last_r)/2)
    cond_E_mean.append(np.mean(E[np.logical_and(R >= last_r, R <= r)]))

    last_r = r

#py.plot([go.Scatter(x= R_vals, y = cond_E_mean)], filename = 'error weights conditional on recurrent')
fig = tools.make_subplots(rows=1, cols=2, print_grid=False, subplot_titles=('All hist', 'Diag hist'))


fig.append_trace(go.Histogram(x=E.flatten()), 1,1)
fig.append_trace(go.Histogram(x=np.diag(E)), 1,2)
py.plot(fig, filename='error weights hist')
