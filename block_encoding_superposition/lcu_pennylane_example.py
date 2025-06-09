#import cudaq
import pennylane as qml
import numpy as np

a = 0.25
b = 0.75

# matrix to be decomposed
A = 2* np.array(
    [[a,  0, 0,  b],
     [0, -a, b,  0],
     [0,  b, a,  0],
     [b,  0, 0, -a]]
)

LCU = qml.pauli_decompose(A)
LCU_coeffs, LCU_ops = LCU.terms()

print(f"LCU decomposition:\n {LCU}")
print(f"Coefficients:\n {LCU_coeffs}")
print(f"Unitaries:\n {LCU_ops}")

dev1 = qml.device("default.qubit", wires=1)

# normalized square roots of coefficients
alphas = (np.sqrt(LCU_coeffs) / np.linalg.norm(np.sqrt(LCU_coeffs)))
print("alphas_norm :", np.linalg.norm(alphas))


#@qml.qnode(dev1)
def prep_circuit(wires):
    qml.StatePrep(alphas, wires=wires)
    return

#@qml.qnode(dev1)
def func(wires):
    prep_circuit(wires=wires)
    return qml.state()

with np.printoptions(precision=3, suppress=True):
    print("Target state: ", alphas)
    print("Output state: ", qml.QNode(func=func, device=dev1)(wires=0))

dev2 = qml.device("default.qubit", wires=3)

# unitaries
ops = LCU_ops
# relabeling wires: 0 → 1, and 1 → 2
unitaries = [qml.map_wires(op, {0: 1, 1: 2}) for op in ops]


#@qml.qnode(dev2)
def sel_circuit():
    qml.Select(unitaries, control=0)
    return

def func2():
    sel_circuit()
    return qml.state()


print(qml.draw(func2, level=99, show_all_wires=True)())

@qml.qnode(dev2)
def lcu_circuit(wires_prep):
    prep_circuit(wires=wires_prep)
    sel_circuit()
    qml.adjoint(prep_circuit)(wires=wires_prep)
    return qml.state()

print(qml.draw(lcu_circuit, level=None, show_all_wires=True)(wires_prep=[0]))

with np.printoptions(precision=3, suppress=True):
    print(qml.matrix(lcu_circuit)(wires_prep=[0]))

init_state = np.array([1,1,1,1])
init_state_norm = np.linalg.norm(init_state)
init_state_normalized = init_state / init_state_norm
@qml.qnode(qml.device("default.qubit", wires=3))
def lcu_circuit2(wires_prep):
    qml.StatePrep(init_state_normalized, wires=[1,2])
    prep_circuit(wires=wires_prep)
    sel_circuit()
    qml.adjoint(prep_circuit)(wires=wires_prep)
    return qml.state()

print(qml.draw(lcu_circuit2, level=None, show_all_wires=True)(wires_prep=[0]))
state_after_lcu = lcu_circuit2(wires_prep=[0])
#state_after_lcu_normalized = state_after_lcu[:4] / np.linalg.norm(state_after_lcu[:4])
state_after_lcu_normalized = state_after_lcu[:4]
with np.printoptions(precision=3, suppress=True):
    print(state_after_lcu[:4]*init_state_norm*np.sum(LCU_coeffs))
print(A @ init_state)


