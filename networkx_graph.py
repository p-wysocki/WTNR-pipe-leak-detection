from networkx.algorithms.shortest_paths.weighted import all_pairs_dijkstra_path_length
import networkx as nx
from dataclasses import dataclass
import matplotlib.pyplot as plt
import wntr_WSN


# constants
unknown_pressure = -1																			# for initialising a Pipe with unknown pressure (pre-simulation)
data_file = wntr_WSN.get_data_file()															# water supply network file
junctions_with_no_edges = ['JPMP51', 'JPMP63', 'JPMP72', 'R5', 'R6', 'R7']						# nodes not connected to anything

# available measurements
flows_temp = ['P544', 'P266', 'P22', 'P43', 'P45', 'P539', 'P460', 'P9', 'P13', 'P536', 'P511', 'P46', 'P52', 'P478', 'P520', 'P27',
              'P280', 'P110', 'P213', 'P459', 'P500', 'P174', 'P244', 'P204', 'P55', 'P56']
pressures_temp = ['J163', 'J244', 'J28', 'J137', 'J52', 'J539', 'J284', 'J12', 'J17', 'J163', 'J43', 'J54', 'J60', 'J39', 'J316',
                  'J33', 'J146', 'J113', 'J151', 'J67', 'J79', 'J172', 'J129', 'J207', 'J62', 'J64']
available_measurements = {'flows': flows_temp, 'pressures': pressures_temp}

# data structures
@dataclass
class Junction:
	name: str
	x: float
	y: float
	pressure: float
	measurement: bool

@dataclass
class Pipe:
	name: str
	node1: str
	node2: str
	length: float
	status: str
	measurement: bool

def get_node_data(data_file: str) -> [list, list]:
	"""
	Parses .inp file and finds data on pipes and junctions, creates Pipe and Junction objects
	Arguments: 	.inp file (str)
	Output: 	[junctions, pipes] where pipes is a list of pipes and junctions is a list of junctions
	"""
	junctions_found = False
	pipes_found = False
	junctions = []
	pipes = []

	with open(data_file, 'r') as file:
		content = iter(file.readlines())														# get all content into single var
		for line in content:
			current_line = line.split()															# get rid of whitespaces

			if current_line == ['[PIPES]']:														# look for pipes info section
					pipes_found = True
					next(content)																# skip a line (column headers)
					continue
			if current_line == ['[COORDINATES]']:												# look for pipes info section
					junctions_found = True
					next(content)																# skip a line (column headers)
					continue

			if pipes_found and not junctions_found:												# if in pipes section
				if current_line == []:															# if end of pipes section
					pipes_found = False															# pipes section ends with \n
					continue
				else:
					name, node1, node2, length, _, _, _, status, _ = current_line
					if name in available_measurements['flows']:									# check if flow measurement is available for this pipe
						is_meas_available = True
					else:
						is_meas_available = False
					pipes.append(Pipe(name=name,
						node1=node1,
						node2=node2,
						length=length,
						status=status,
						measurement=is_meas_available))

			if junctions_found and not pipes_found:												# if in junctions section
				if current_line == []:															# if end of junctions section
					junctions_found = False														# junctions section ends with \n
					continue
				else:
					name, x, y = current_line
					if name in available_measurements['pressures']:								# check if flow measurement is available for this pipe
						is_meas_available = True
					else:
						is_meas_available = False
					junctions.append(Junction(name=name,
						x=x,
						y=y,
						pressure=unknown_pressure,												# initialise an unknown pressure
						measurement=is_meas_available))															
	return [junctions, pipes]

def create_graph() -> nx.classes.graph.Graph:
	"""
	Creates a NetworkX graph object from lists of nodes and edges taken from .inp file
	Arguments: 	nodes - list of Junction() objects
				edges - list of Pipe() objects
	Returns:	NetworkX graph (networkx.classes.graph.Graph)
	"""

	print("Creating a NetworkX graph from .inp file (networkx_graph.py - create_graph())")

	# get data from .inp file
	nodes, edges = get_node_data(data_file)

	# remove junctions with no edges
	nodes = [i for i in nodes if i.name not in junctions_with_no_edges]

	# create a graph object
	G = nx.Graph()

	# add junctions (nodes), specify their coordinates
	for node in nodes:
		G.add_node(node.name, pos=(float(node.x), float(node.y)), measurement=node.measurement)

	# add pipes (edges)
	for edge in edges:
		G.add_edge(edge.node1, edge.node2,
			weight=float(edge.length), measurement=edge.measurement, name=edge.name)

	return G 																					# return nx.classes.graph.Graph

def get_closest_sensors(G: nx.classes.graph.Graph, central_node: str, n: any, shortest_paths: list) -> [list, list]:
	"""
	Creates a list of sensors closest to the main node (both pressure and flow)
				using NetworkX all_pairs_dijkstra_path_length()
	Arguments:	G - NetworkX graph of the water network
				central_node - name of junction you need sensors positions relative to
				n - number of closest sensors, int or str ('all' for all sensors)
				shortest_paths - list(all_pairs_dijkstra_path_length(G)) from NetworkX
	Returns: 	[pressure_sensors, flow_sensors]
				both lists of n closest [sensor_name, distance_from_main_node]
				both sorted by distance (ascending), or all sensors if n='all'
	"""
	#print("Calculationg sensors closest to main node (networkx_graph.py - get_closest_sensors())")
	nodes_measured = []																			# which junctions have pressure sensors in them?
	edges_measured = []																			# which pipes have flow sensors?

	# make lists of node/edge objects with sensors in them
	for node in G.nodes(data=True):
	    if node[1]['measurement'] is True:
	        nodes_measured.append(node)
	for edge in G.edges(data=True):
	    if edge[2]['measurement'] is True:
	        edges_measured.append(edge)

	objects_measured = {'junctions': nodes_measured, 'pipes': edges_measured}					# sum it up in one var

	# find our central_node in weird all_pairs_dijkstra_path_length(G) format
	for centrum in shortest_paths:
		if centrum[0] == central_node:
			shortest_paths_from_central_node = centrum

    # get a list of pressure sensors with their respective distances from central_node
	pressure_sensors_with_distance = []
	for node in objects_measured['junctions']:
		pressure_sensors_with_distance.append((node[0], shortest_paths_from_central_node[1][node[0]]))

	# do the same for flow sensors
	flow_sensors_with_distance = []
	for edge in objects_measured['pipes']:
		flow_sensors_with_distance.append((edge[2]['name'], shortest_paths_from_central_node[1][edge[0]]))

	# sort them by distance (ascending)
	pressure_sensors_sorted_by_distance = sorted(pressure_sensors_with_distance, key=lambda sensor: sensor[1])
	flow_sensors_sorted_by_distance = sorted(flow_sensors_with_distance, key=lambda sensor: sensor[1])

	if n == 'all':
		return [pressure_sensors_sorted_by_distance, flow_sensors_sorted_by_distance]			# return all sensors
	else:
		return [pressure_sensors_sorted_by_distance[:n], flow_sensors_sorted_by_distance[:n]]	# return only n closest sensors

def get_sensor_names() -> dict:
	"""
	Returns names of available sensors names (dict)
	Purpose is to get the value to another .py file
	"""
	return available_measurements

if __name__ == '__main__':

	G = create_graph()
	shortest_paths = list(all_pairs_dijkstra_path_length(G))
	a, b = get_closest_sensors(G, 'J63', 'all', shortest_paths)
	print(b)
	#nx.draw(G, nx.get_node_attributes(G, 'pos'), node_size=20)
	#length = wn.query_link_attribute('length')
	#G = wn.get_graph(wn, link_weight=length)
	#plt.show()

