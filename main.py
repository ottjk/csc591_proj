import pennylane as qml
from pennylane import numpy as np
from rich.progress import track

n_vertices = 5
n_wires = 2*n_vertices
n_layers = 3

graph = [(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(2,3),(2,4),(3,4)]

cost_hamiltonian = sum(qml.Hamiltonian([1,1,1], [
    qml.PauliZ(2*edge[0]) @ qml.PauliZ(2*edge[1]),
    qml.PauliZ(2*edge[0]+1) @ qml.PauliZ(2*edge[1]+1),
    qml.PauliZ(2*edge[0]) @ qml.PauliZ(2*edge[0]+1) @ qml.PauliZ(2*edge[1]) @ qml.PauliZ(2*edge[1]+1)
]) for edge in graph)

# unitary operator U_B with parameter beta
def U_B(beta):
    for wire in range(n_wires):
        qml.RX(2 * beta, wires=wire)

# unitary operator U_C with parameter gamma
def U_C(gamma):
    for edge in graph:
        wire1 = 2*edge[0]
        wire2 = wire1+1
        wire3 = 2*edge[1]
        wire4 = wire3+1

        qml.CNOT(wires=[wire1, wire3])
        qml.RZ(gamma, wires=wire3)
        qml.CNOT(wires=[wire1, wire3])

        qml.CNOT(wires=[wire2, wire4])
        qml.RZ(gamma, wires=wire4)
        qml.CNOT(wires=[wire2, wire4])

        qml.CNOT(wires=[wire1, wire4])
        qml.CNOT(wires=[wire2, wire4])
        qml.CNOT(wires=[wire3, wire4])
        qml.RZ(gamma, wires=wire4)
        qml.CNOT(wires=[wire3, wire4])
        qml.CNOT(wires=[wire2, wire4])
        qml.CNOT(wires=[wire1, wire4])

def circuit(gammas, betas):
    for wire in range(n_wires):
        qml.Hadamard(wires=wire)

    for i in range(n_layers):
        U_C(gammas[i])
        U_B(betas[i])

dev = qml.device("default.qubit", wires=n_wires)

@qml.qnode(dev)
def cost_function(params):
    circuit(params[0], params[1])
    return qml.expval(cost_hamiltonian)

@qml.qnode(dev)
def probability_circuit(params):
    circuit(params[0], params[1])
    return qml.probs()

def qaoa_color():
    params = np.random.rand(2, n_layers, requires_grad=True)
    print(params)

    opt = qml.QNSPSAOptimizer()

    steps = 200
    for i in track(range(steps)):
        params, cost = opt.step_and_cost(cost_function, params)
        probs = probability_circuit(params)
        print("{: .7f}, {:010b}".format(cost, np.argmax(probs)))

    return params

@qml.qnode(dev)
def h_exp():
    return qml.expval(cost_hamiltonian)

params = qaoa_color()
