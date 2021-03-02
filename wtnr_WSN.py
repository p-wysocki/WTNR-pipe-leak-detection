import wntr
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

def get_sim_results() -> wntr.sim.results.SimulationResults:
	inp_file = 'Walkerton_v1.inp'
	wn = wntr.network.WaterNetworkModel(inp_file)
	sim = wntr.sim.WNTRSimulator(wn)
	return sim.run_sim()

def get_water_network_object(weighted = False: bool) -> networkx.classes.multidigraph.MultiDiGraph:
	
	return 

if __name__ == '__main__':
	inp_file = 'Walkerton_v1.inp'
	wn = wntr.network.WaterNetworkModel(inp_file)
	#sim = wntr.sim.WNTRSimulator(wn)
	#results = sim.run_sim()
	length = wn.query_link_attribute('length')
	G = wn.get_graph(wn, link_weight=length)
	nx.draw(G)

