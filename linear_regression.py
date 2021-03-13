from networkx.algorithms.shortest_paths.weighted import all_pairs_dijkstra_path_length
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import wntr
import wntr_WSN
import pandas as pd
import networkx_graph as ng
import sklearn
from sklearn.model_selection import train_test_split

def model_network_with_linreg(n: any) -> list:
	"""
	Model every node in the water network with an sklearn.linear_regression() model based on readings from 
	n closest sensors
	Arguments:	n - amount of closest sensors to read data from. Uses n of each, eg. if n=1 readings from the closest
				flowmeter AND the closest pressure meter are used (int or str 'all')
	Returns:	list of dicts describing each node in the network {'node', 'regression', 'msq', 'r2'}
															where:	node - node name (str)
																	regression - sklearn regression object
																	msq - mean squared error (float)
																	r2 - R2 score (float)
																	sensors - list of strings, names of sensors used
	"""
	print("Modeling the network with linreg (linear_regression.py - model_network_with_linregression())")

	# collect data
	G = ng.create_graph()																							# get NetworkX graphs
	amount_of_junctions = len(G.nodes())																			# for printing progress
	amount_of_pipes = len(G.edges())																				# for printing progress
	shortest_paths = list(all_pairs_dijkstra_path_length(G))														# get shortest distances between all node pairs
	sim_results = wntr_WSN.get_sim_results()																		# no leak simulation results
	all_pressure_readings, all_flowrate_readings = get_all_sensor_readings(sim_results)								# extract readings for sensors we have

	# preprocess X data
	data_X = pd.concat([all_pressure_readings, all_flowrate_readings], axis=1, join='inner')						# join readings into a single pandas DF
	data_X = data_X.loc[:, ~data_X.columns.duplicated()]															# remove duplicate columns
	normalization_params = {'mean': data_X.mean(), 'std': data_X.std()}												# collect normalization params
	data_X = (data_X - normalization_params['mean']) / normalization_params['std']									# normalise X set

	# for each node in graph create a separate sklearn linear_regression() object
	linreg_models = []
	for i, node in enumerate(G.nodes()):
		print(f'\tModeling junction {node}: ({i+1} out of {amount_of_junctions})')									# printing progress
		data_X_scope = data_X 																						# copy of data_X inside for scope
		sensors = ng.get_closest_sensors(G=G, 																		# retrieve sensors closest to node
										 central_node=node,
										 n=n,																		# nr of sensors 
										 shortest_paths=shortest_paths)
		sensors_names = [i[0] for i in sensors]																		# and flowrate readings, get rid of distances

		data_X_scope = data_X_scope[sensors_names]																	# extract data for the closest sensors

		data_y = sim_results.node['pressure'][node]																	# get y data for supervised learning
		train_X, test_X, train_y, test_y = train_test_split(data_X_scope,											# split and shuffle dataset
												 		  	data_y,
												 		  	test_size=0.1)

		linreg_models.append(create_linreg_model(node, train_X, test_X, train_y, test_y))							# model the node and add it to the list

	return linreg_models, normalization_params

def get_all_sensor_readings(sim_results: wntr.sim.results.SimulationResults)\
 -> [pd.core.frame.DataFrame, pd.core.frame.DataFrame]:
	"""
	Retrieve flowrate and pressure readings of all sensors
	throughout the whole simulation time
	Arguments:	WNTR simulation results object (wntr.sim.results.SimulationResults)
	Returns:	list of two pandas DataFrames [pressure_readings, flowrate_readings]
	"""

	# retrieve flowrate and pressure sensors names
	sensors = ng.get_sensor_names()
	pressure_sensors = sensors['pressures']
	flowrate_sensors = sensors['flows']

	# get sim results for all nodes
	all_pressure_results = sim_results.node['pressure']
	all_flowrate_results = sim_results.link['flowrate']

	# leave only data we can measure by sensors
	pressure_readings = all_pressure_results[pressure_sensors]
	flowrate_readings = all_flowrate_results[flowrate_sensors]

	return [pressure_readings, flowrate_readings]

def create_linreg_model(node, train_X, test_X, train_y, test_y) -> dict:
	"""
	Creates sklearn linear regression object for a given node
	Arguments:	node - node name (str)
				train_X, test_X - (DF)
				train_y, test_y - (series)
	Returns:	dict 	{'node', 'regression', 'msq', 'r2'}
						where:	node - node name (str)
								regression - sklearn regression object
								msq - mean squared error (float)
								r2 - R2 score (float)
								sensors - list of strings, names of sensors used
	"""
	# create model and fit the data
	regression = linear_model.LinearRegression()
	regression.fit(train_X, train_y)

	# gather predictions on test set
	test_pred = regression.predict(test_X)
	
	# evaluate the model
	msq = mean_squared_error(test_y, test_pred)
	r2 = r2_score(test_y, test_pred)

	return {'node': node, 'linreg': regression, 'msq': msq, 'r2': r2, 'sensors': train_X.columns.tolist()}

def predict_pressure_on_leaks(linreg_models: list, norm_params: dict, leak_start = 10, leak_end = 40, leak_area = 0.0001) -> pd.core.frame.DataFrame:
	"""
	For every modeled node in the network simulate hydraulics with it leaking
	Arguments:	linreg_models - list of dicts from model_network_with_linreg()
				norm_params - dict of parameters, also from model_network_with_linreg()
				leak_start - leak start time in hours (int)
				leak_end - leak end time in hours (int)
				leak_area - area of the leak (meters squared)
	Returns:	DataFrame where each node has its separate column with pressures predicted for a scenario of its leak, index is the time in seconds
	"""
	G = ng.create_graph()
	leak_predictions = {}
	temp = 0#TEMP simulate leak only for first 10 nodes, to save time in debugging---------------------------------------------------------
	# for each node simulate its leak, and predict its pressure using its linear regression model
	for node in G.nodes():
		print(f'Simulating node {node} with leak')
		temp += 1#TEMP simulate leak only for first 10 nodes, to save time in debugging---------------------------------------------------------
		if temp>10:#TEMP simulate leak only for first 10 nodes, to save time in debugging---------------------------------------------------------
			break#TEMP simulate leak only for first 10 nodes, to save time in debugging---------------------------------------------------------
		try:
			# simulate hydraulics with a leak on this particular node
			sim_results_L = wntr_WSN.get_sim_results_LEAK(node=node, area=leak_area, start_time=leak_start, end_time=leak_end)

			# find linear regression model for this particular node
			for model in linreg_models:
				if model['node'] == node:
					linreg_dict = model

			# join flowrate and pressure measurements into a single DataFrame
			sim_results_L = pd.concat([sim_results_L.node['pressure'], sim_results_L.link['flowrate']], axis=1)

			# choose only the available sensors
			sim_results_L = sim_results_L[linreg_dict['sensors']]

			# choose only leak timeframe (index values in seconds)
			sim_results_L = sim_results_L.loc[[3600*i for i in range(leak_start, leak_end+1)]]
			
			# normalise the input data with respective parameters
			sim_results_L = (sim_results_L - norm_params['mean'].loc[linreg_dict['sensors']]) / norm_params['std'].loc[linreg_dict['sensors']]

			# predict pressure for each node
			leak_predictions[node] = linreg_dict['linreg'].predict(sim_results_L)

		except Exception as e:
			# sometimes an internal WNTR error happens, I didn't manage to find out why (function convergence error)
			print(f'Runtime error on simulation of node {node} with leak (convergence failed)\nError message:\n{e}')

	# create a DataFrame of results, index is time of leak in seconds
	index = [3600*i for i in range(leak_start, leak_end+1)]
	results = pd.DataFrame(data=leak_predictions, index=index)
	return results

def find_residuals(pressure_predictions_L: pd.core.frame.DataFrame, leak_start: int, leak_end: int) -> pd.core.frame.Series:
	"""
	For every modeled node calculate residuum used for leak detection. Residuum is a diffrence between:
				y - actual pressure values coming from a simulation WITHOUT leaks
				y_hat - values predicted by linear regression models using data from simulations WITH leaks
				The function calculates diffrences in leak timeframe and returns the minimal value as a residual
	Arguments:	pressure_predictions_L - y mentioned above, from predict_pressure_on_leaks()
				leak_start - leak start time in hours (int)
				leak_end - leak end time in hours (int)
	Returns:	pd.core.frame.Series - Series of residuals for each modeled node
	"""
	G = ng.create_graph()

	# get simulation results for the network WITHOUT leaks
	sim_results_nL = wntr_WSN.get_sim_results()

	# select the leak timeframe (index values in seconds)
	nodes_pressure_nL = sim_results_nL.node['pressure'].loc[[3600*i for i in range(leak_start, leak_end+1)]]

	# select columns corresponding to nodes that have predictions ready (to subtract DataFrames cleanly)
	nodes_pressure_nL = nodes_pressure_nL[pressure_predictions_L.columns.tolist()]

	# get absolute diffrence, then choose the minima for each node (column)
	results = (nodes_pressure_nL - pressure_predictions_L).abs().min(axis=0)
	return results

def check_network_for_leaks():
	pass

if __name__ == '__main__':
	leak_start = 10
	leak_end = 40
	leak_area = 0.0005

	models, normalization_params = model_network_with_linreg(n='all')
	pressure_predicted_L = predict_pressure_on_leaks(linreg_models=models, norm_params=normalization_params,
													 leak_start=leak_start, leak_end=leak_end, leak_area=leak_area)
	residuals = find_residuals(pressure_predictions_L=pressure_predicted_L, leak_start=leak_start, leak_end=leak_end)
	print(residuals)