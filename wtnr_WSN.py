import wntr
import pandas as pd

def get_sim_results():
	inp_file = 'Walkerton_v1.inp'
	wn = wntr.network.WaterNetworkModel(inp_file)
	sim = wntr.sim.WNTRSimulator(wn)
	return sim.run_sim()
