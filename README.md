# WTNR-pipe-leak-detection
Machine learning university research project focused on detecting underground equipment failure

The analysed algorithm models all of water networks' nodes with linear regression using a limited set of sensors. They are chosen by Dijkstra path finding algorithm (closest sensors weighted by the pipe length). The water network model of Walkerton city is contained in a file of .inp extension which I am not allowed to share.

The files:
1) linear_regression.ipynb           - file used when getting familiarised with the WNTR library, and prototyping first linreg models
2) networkx_graph.py                 - handles translating .inp file into NetworkX format and generating NetworkX graphs, as well as Dijkstra pathing
3) wntr_WSN.py                       - handles functionality related to WNTR library, mainly simulating the hydraulics with and without leaks
4) linear_regression.py              - handles machine learning (linear regression modeling, residuum finding, algorithm evaluation functions)
5) simulations_to_csv.ipynb          - WNTR simulations take lots of time, so they need to be saved locally to .csv files and then read
6) dashboard.py                      - interactive web dashboard showing the water network with readings/linreg predictions
7) final_output.ipynb                - used for creating standarised final output files for further analysis
