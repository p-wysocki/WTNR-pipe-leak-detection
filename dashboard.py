import pandas as pd
import plotly.express as px  # (version 4.7.0)
import plotly.graph_objects as go
import networkx as nx
from dataclasses import dataclass
import matplotlib.pyplot as plt
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import networkx_graph as ng 																		# local file network_graph.py
import wntr_WSN as WSN 																				# local file WTNR Water Supply Network

# https://towardsdatascience.com/python-interactive-network-visualization-using-networkx-plotly-and-dash-e44749161ed7
# https://www.youtube.com/watch?v=hSPmj7mK6ng

app = dash.Dash(__name__)
sim_results = WSN.get_sim_results()
G = ng.create_graph()

# create nodes/edges
edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = G.nodes[edge[0]]['pos']
    x1, y1 = G.nodes[edge[1]]['pos']
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#000000'),
    showlegend=False,
    hoverinfo='none',
    mode='lines')

node_x = []
node_y = []
for node in G.nodes():
    x, y = G.nodes[node]['pos']
    node_x.append(x)
    node_y.append(y)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    visible=True,
    showlegend=False,
    marker=dict(
        showscale=True,
        # colorscale options
        #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
        #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
        colorscale='RdBu',
        reversescale=True,
        color=[],
        size=10,
        colorbar=dict(
            thickness=15,
            title='Pressure at junctions [m]',
            xanchor='left',
            titleside='right'
        ),
        line_width=2))

# coloring
pressure_measurements = []
node_names = []
for node in G.nodes():
	# get data for t=0 as default
    pressure_measurements.append(sim_results.node['pressure'].loc[0].loc[node])
    node_names.append(node + ' - Pressure: ' + str(round(sim_results.node['pressure'].loc[0].loc[node], 2)) + 'm')

node_trace.marker.color = pressure_measurements
node_trace.text = node_names

fig = go.Figure(data=[edge_trace, node_trace],
	             layout=go.Layout(
	                #title='<br>...',
	                titlefont_size=16,
	                height=801,
	                showlegend=False,
	                hovermode='closest',
	                margin=dict(b=20, l=5, r=5, t=40),
	                annotations=[ dict(
	                    text="Author: Przemyslaw Wysocki, Faculty of Mechatronics at Warsaw University of Technology",
	                    showarrow=False,
	                    xref="paper", yref="paper",
	                    x=0.005, y=-0.002 ) ],
	                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
	                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
	                )

app.layout = html.Div([

    html.H1("Water Supply Network", style={'text-align': 'center'}),
    html.H2("Choose simulation time on slider below (0 - 100h)", style={'text-align': 'center'}),

    dcc.Slider(
    id='slider_value',
    min=0,
    max=100,
    step=None,
    marks={i: str(i) for i in range(101)},
    value=0
),

	dcc.Graph(id='network_graph', figure=fig),

	#html.Div(id='output_container1', children=[]),

])

@app.callback(
    Output(component_id='network_graph', component_property='figure'),
    [Input(component_id='slider_value', component_property='value')]
)
def update_graph(hrs_elapsed):
	pressure_measurements = []
	node_names = []

	# re-get water pressure from results DataFrame
	for node in G.nodes():
		node_names.append(node + ' - Pressure: ' + str(round(sim_results.node['pressure'].loc[hrs_elapsed*3600].loc[node], 2)) + 'm')
		pressure_measurements.append(sim_results.node['pressure'].loc[hrs_elapsed*3600].loc[node])

	node_trace.marker.color = pressure_measurements
	node_trace.text = node_names

	# update the figure
	return go.Figure(data=[edge_trace, node_trace])

if __name__ == '__main__':
    app.run_server(debug=True)
