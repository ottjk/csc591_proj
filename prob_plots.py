from main import *

import matplotlib
matplotlib.use('module://matplotlib-backend-sixel')
import matplotlib.pyplot as plt

n_layers = 4

def get_probs(qc):
    print(f'Finding probabilities for {qc.n_vertices}-vertex graph')
    params = qc.qaoa_color()
    probs = qc.probability_circuit(params)
    print(f"{len(probs[probs > max(probs) * 3/4])}")

    plot = plt.bar(range(2 ** qc.n_wires), probs)
    plt.savefig(f'figures/{qc.n_vertices}-node-probs.pdf')

n_vertices = 5
graph = [(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(2,3),(2,4),(3,4)]
qc_5 = QC(graph, n_vertices, n_layers)

n_vertices = 6
graph = [(0,1),(0,2),(0,3),(0,4),(0,5),(1,2),(2,3),(3,4),(4,5),(5,1)]
qc_6 = QC(graph, n_vertices, n_layers)

# get_probs(qc_5)
get_probs(qc_6)
