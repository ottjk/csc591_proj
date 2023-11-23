import pennylane as qml
from pennylane import numpy as np

n_vertices = 5
n_wires = 2*n_vertices

graph = [(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(2,3),(2,4),(3,4)]
# graph = [(0,2)]

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


def bitstring_to_int(bit_string_sample):
    bit_string = "".join(str(bs) for bs in bit_string_sample)
    return int(bit_string, base=2)

dev = qml.device("default.qubit", wires=n_wires, shots=1)

@qml.qnode(dev)
def circuit(gammas, betas, sample=False, n_layers=1):
    # apply Hadamards to get the n qubit |+> state
    for wire in range(n_wires):
        qml.Hadamard(wires=wire)
    # p instances of unitary operators
    for i in range(n_layers):
        U_C(gammas[i])
        U_B(betas[i])
    if sample:
        # measurement phase
        return qml.sample()

    return qml.expval(cost_hamiltonian)

def qaoa_maxcut(n_layers=1):
    print("\np={:d}".format(n_layers))

    # initialize the parameters near zero
    init_gammas = 0.01 * np.random.rand(n_layers, requires_grad=True)
    init_betas = 0.01 * np.random.rand(n_layers, requires_grad=True)

    # initialize optimizer: Adagrad works well empirically
    opt = qml.QNSPSAOptimizer()

    # optimize parameters in objective
    gammas = init_gammas
    betas = init_betas
    steps = 200
    for i in range(steps):
        params, cost = opt.step_and_cost(circuit, gammas, betas, n_layers=n_layers)
        gammas = params[0]
        betas = params[1]
        if (i + 1) % 5 == 0:
            print("Objective after step {:5d}: {: .7f}".format(i + 1, cost))

    # sample measured bitstrings 100 times
    bit_strings = []
    n_samples = 1000
    for i in range(0, n_samples):
        bit_strings.append(bitstring_to_int(circuit(gammas, betas, sample=True, n_layers=n_layers)))

    # print optimal parameters and most frequently sampled bitstring
    counts = np.bincount(np.array(bit_strings))
    most_freq_bit_string = np.argmax(counts)
    print("Optimized (gamma, beta) vectors:\n{} {}".format(gammas, betas))
    print("Most frequently sampled bit string is: {:010b}".format(most_freq_bit_string))

    return circuit(gammas, betas, n_layers=n_layers), bit_strings

bitstrings1 = qaoa_maxcut(n_layers=2)[1]

# import matplotlib.pyplot as plt
# matplotlib.use('module://matplotlib-backend-sixel')
#
# xticks = range(0, 16)
# xtick_labels = list(map(lambda x: format(x, "04b"), xticks))
# bins = np.arange(0, 17) - 0.5
#
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
# plt.subplot(1, 2, 1)
# plt.title("n_layers=1")
# plt.xlabel("bitstrings")
# plt.ylabel("freq.")
# plt.xticks(xticks, xtick_labels, rotation="vertical")
# plt.hist(bitstrings1, bins=bins)
# plt.subplot(1, 2, 2)
# plt.title("n_layers=2")
# plt.xlabel("bitstrings")
# plt.ylabel("freq.")
# plt.xticks(xticks, xtick_labels, rotation="vertical")
# plt.hist(bitstrings2, bins=bins)
# plt.tight_layout()
# plt.show()
