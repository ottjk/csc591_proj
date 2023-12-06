import pennylane as qml
from pennylane import numpy as np
from rich.progress import track

def generate_hamiltonian(graph):
    cost_hamiltonian = sum(qml.Hamiltonian([1/4,1/4,1/4], [
        qml.PauliZ(2*edge[0]) @ qml.PauliZ(2*edge[1]),
        qml.PauliZ(2*edge[0]+1) @ qml.PauliZ(2*edge[1]+1),
        qml.PauliZ(2*edge[0]) @ qml.PauliZ(2*edge[0]+1) @ qml.PauliZ(2*edge[1]) @ qml.PauliZ(2*edge[1]+1)
    ]) for edge in graph)
    return cost_hamiltonian

class QC:
    def __init__(self, graph, n_vertices, n_layers):
        self.graph = graph
        self.n_vertices = n_vertices
        self.n_wires = 2*n_vertices
        self.n_layers = n_layers
        self.hamiltonian = generate_hamiltonian(graph)
        self.dev = qml.device("default.qubit", wires=self.n_wires)

    # unitary operator U_B with parameter beta
    def U_B(self, beta):
        for wire in range(self.n_wires):
            qml.RX(beta, wires=wire)

    # unitary operator U_C with parameter gamma
    def U_C(self, gamma):
        for edge in self.graph:
            wire1 = 2*edge[0]
            wire2 = wire1+1
            wire3 = 2*edge[1]
            wire4 = wire3+1

            qml.CNOT(wires=[wire1, wire3])
            qml.RZ(gamma/4, wires=wire3)
            qml.CNOT(wires=[wire1, wire3])

            qml.CNOT(wires=[wire2, wire4])
            qml.RZ(gamma/4, wires=wire4)
            qml.CNOT(wires=[wire2, wire4])

            qml.CNOT(wires=[wire1, wire4])
            qml.CNOT(wires=[wire2, wire4])
            qml.CNOT(wires=[wire3, wire4])
            qml.RZ(gamma/4, wires=wire4)
            qml.CNOT(wires=[wire3, wire4])
            qml.CNOT(wires=[wire2, wire4])
            qml.CNOT(wires=[wire1, wire4])

    def circuit(self, gammas, betas):
        for wire in range(self.n_wires):
            qml.PauliX(wire)
            qml.Hadamard(wires=wire)

        for i in range(self.n_layers):
            self.U_C(gammas[i])
            self.U_B(betas[i])
    
    def cost_function(self):
        @qml.qnode(self.dev)
        def _cost_function(params):
            self.circuit(params[0], params[1])
            return qml.expval(self.hamiltonian)
        return _cost_function

    def probability_circuit(self, params):
        @qml.qnode(self.dev)
        def _probability_circuit(params):
            self.circuit(params[0], params[1])
            return qml.probs()
        return _probability_circuit(params)

    def h_exp(self):
        @qml.qnode(self.dev)
        def _h_exp():
            return qml.expval(self.hamiltonian)
        return _h_exp()

    def qaoa_color(self):
        gammas = np.linspace(0, 1, num=self.n_layers)
        betas = np.linspace(1, 0, num=self.n_layers)
        params = np.array([gammas,betas])

        opt = qml.QNSPSAOptimizer()

        steps = 10
        for i in track(range(steps)):
            params, cost = opt.step_and_cost(self.cost_function(), params)
            probs = self.probability_circuit(params)

            print(f"{cost: .7f}, {np.argmax(probs):0{self.n_wires}b}")

        return params
