import wntr
import wntr_WSN
import pandas as pd
import networkx_graph as ng


def model_network_with_linregression():
	G = ng.create_graph()
	sim_results = wntr_WSN.get_sim_results()
	pressure_readings, flowrate_readings = get_sensor_readings(sim_results)
	learning_data = pd.concat([pressure_readings, flowrate_readings], axis=1, join='inner')						# join into a single pandas DF
	return

def get_sensor_readings(sim_results: wntr.sim.results.SimulationResults)\
 -> [pd.core.frame.DataFrame, pd.core.frame.DataFrame]:
	"""
	Retrieve flowrate and pressure readings of available sensors
	throughout the whole simulation time
	Arguments:	simulation results object (wntr.sim.results.SimulationResults)
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

def create_lin_regression_model(learning_data):
	pass

def check_network_for_leaks():
	pass

if __name__ == '__main__':
	model_network_with_linregression()