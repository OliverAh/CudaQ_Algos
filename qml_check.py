import pennylane as qml

dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def circuit():
    qml.Hadamard(wires=1)
    return qml.state()

print(qml.draw(circuit, show_all_wires=True)())
print(circuit())