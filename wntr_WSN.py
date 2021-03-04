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
	WORK IN PROGRESS
	Runs and returns a hydraulic simulation WNTR object  
	"""
	print("Running the simulation (wntr_WSN.py - get_sim_results())")
	wn = wntr.network.WaterNetworkModel(data_file)
	sim = wntr.sim.WNTRSimulator(wn)
	return sim.run_sim()

def simulate_with_leak():
	pass

if __name__ == '__main__':
	inp_file = 'Walkerton_v1.inp'
	wn = wntr.network.WaterNetworkModel(data_file)
	#sim = wntr.sim.WNTRSimulator(wn)
	#results = sim.run_sim()
	length = wn.query_link_attribute('length')
	G = wn.get_graph(wn, link_weight=length)
	nx.draw(G)

