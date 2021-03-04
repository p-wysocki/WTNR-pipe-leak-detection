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
	normalization_params = {'mean': data_X.mean(), 'std': data_X.std()}												# collect normalization params
	data_X = (data_X - normalization_params['mean']) / normalization_params['std']									# normalise X set

	# for each node in graph create a separate sklearn linear_regression() object
	linreg_models = []
	for i, node in enumerate(G.nodes()):
		print(f'\tModeling junction {node}: ({i+1} out of {amount_of_junctions})')									# printing progress
		data_X_scope = data_X 																						# copy of data_X inside for scope
		pressure_sensors, flowrate_sensors = ng.get_closest_sensors(G=G, 											# retrieve sensors closest to node
																	central_node=node,
																	n=n,
																	shortest_paths=shortest_paths)
		available_sensors = pressure_sensors + flowrate_sensors														# combine two lists (data_X has both pressure
		available_sensors_names = [i[0] for i in available_sensors]													# and flowrate readings, get rid of distances

		data_X_scope = data_X_scope[available_sensors_names]														# extract data for the closest sensors

		data_y = sim_results.node['pressure'][node]																	# get y data for supervised learning
		train_X, test_X, train_y, test_y = train_test_split(data_X_scope,											# split and shuffle dataset
												 		  	data_y,
												 		  	test_size=0.1)

		linreg_models.append(create_linreg_model(node, train_X, test_X, train_y, test_y))

	return linreg_models

def get_all_sensor_readings(sim_results: wntr.sim.results.SimulationResults)\
 -> [pd.core.frame.DataFrame, pd.core.frame.DataFrame]:
	"""
	Retrieve flowrate and pressure readings of all sensors
	throughout the whole simulation time
	Arguments:	WNTR simulation results object (wntr.sim.results.SimulationResults)
	Returns:	list of two pandas DataFrames [pressure_readings, flowrate_readings]
	"""
	print("Retrieveing all sensor readings (linear_regression.py - get_all_sensor_readings())")

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
	"""
	# create model and fit the data
	regression = linear_model.LinearRegression()
	regression.fit(train_X, train_y)

	# gather predictions on test set
	test_pred = regression.predict(test_X)
	
	# evaluate the model
	msq = mean_squared_error(test_y, test_pred)
	r2 = r2_score(test_y, test_pred)

	return {'node': node, 'linreg': regression, 'msq': msq, 'r2': r2}

def check_network_for_leaks():
	pass

if __name__ == '__main__':

	model_network_with_linreg(n=1)
