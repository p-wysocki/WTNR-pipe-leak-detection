import networkx as nx
from dataclasses import dataclass
import matplotlib.pyplot as plt


# constants
unknown_pressure = -1																			# for initialising a Pipe with unknown pressure (pre-simulation)
data_file = 'Walkerton_v1.inp'																	# water supply network file
junctions_with_no_edges = ['JPMP51', 'JPMP63', 'JPMP72', 'R5', 'R6', 'R7']						# nodes not connected to anything,

# data structures
@dataclass
class Junction:
	name: str
	x: float
	y: float
	pressure: float

@dataclass
class Pipe:
	name: str
	node1: str
	node2: str
	status: str
		
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
					next(content)																# skip column headers
					continue
			if current_line == ['[COORDINATES]']:												# look for pipes info section
					junctions_found = True
					next(content)																# skip column headers
					continue

			if pipes_found and not junctions_found:												# if in pipes section
				if current_line == []:															# if end of pipes section
					pipes_found = False
					continue
				else:
					name, node1, node2, _, _, _, _, status, _ = current_line
					pipes.append(Pipe(name=name,
						node1=node1,
						node2=node2,
						status=status))

			if junctions_found and not pipes_found:												# if in junctions section
				if current_line == []:															# if end of junctions section
					junctions_found = False
					continue
				else:
					name, x, y = current_line
					junctions.append(Junction(name=name,
						x=x,
						y=y,
						pressure = -1))															# initialise an unknown pressure
	return [junctions, pipes]

def create_graph() -> nx.classes.graph.Graph:
	"""
	Created a NetworkX graph object from lists of nodes and edges taken from .inp file
	Arguments: 	nodes - list of Junction() objects
				edges - list of Pipe() objects
	Returns:	NetworkX graph (networkx.classes.graph.Graph)
	"""

	# get data from .inp file
	nodes, edges = get_node_data(data_file)

	# remove junctions with no edges
	nodes = [i for i in nodes if i.name not in junctions_with_no_edges]

	# create a graph object
	G = nx.Graph()

	# add junctions (nodes), specify their coordinates
	for node in nodes:
		G.add_node(node.name, pos=(float(node.x), float(node.y)))

	# add pipes (edges)
	for edge in edges:
		G.add_edge(edge.node1, edge.node2)

	return G


if __name__ == '__main__':

	G = create_graph()
	nx.draw(G, nx.get_node_attributes(G, 'pos'), node_size=20)
	plt.show()