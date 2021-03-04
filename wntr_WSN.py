import wntr
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


data_file = 'Walkerton_v1.inp'																# water supply network .inp file

def get_data_file() -> str:
	"""
	Returns name of .inp file (str)
	Purpose is to get the value to another .py file
	"""
	return data_file																	

def get_sim_results() -> wntr.sim.results.SimulationResults:
	"""
	Runs and returns a hydraulic simulation WNTR object (NO LEAKS)
	Returns:	Simulation results (wntr.sim.results.SimulationResults)
	"""
	print("Running the simulation (wntr_WSN.py - get_sim_results())")
	wn = wntr.network.WaterNetworkModel(data_file)
	sim = wntr.sim.WNTRSimulator(wn)
	return sim.run_sim()

def get_sim_results_LEAK(node: str, area: float, start_time: int, end_time: int):
	wn_leaks = wntr.network.WaterNetworkModel(data_file)
	leak_node = wn_leaks.get_node(node)
	leak_node.add_leak(wn_leaks, area=area, start_time=start_time*3600, end_time=end_time*3600)
	sim_leaks = wntr.sim.WNTRSimulator(wn_leaks)
	return sim_leaks.run_sim()

if __name__ == '__main__':
	inp_file = 'Walkerton_v1.inp'
	wn = wntr.network.WaterNetworkModel(data_file)
	#sim = wntr.sim.WNTRSimulator(wn)
	#results = sim.run_sim()
	#length = wn.query_link_attribute('length')
	#G = wn.get_graph(wn, link_weight=length)
	#nx.draw(G)
	res = get_sim_results_LEAK('J63', 0.001, 3, 5)
	print(res.node['pressure'])

