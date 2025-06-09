import cudaq
import numpy as np
import pennylane as qml




def _convert_biglittle_endian_unitary(unitary: np.ndarray) -> np.ndarray:
    """
    Converts the unitary matrix of a multiqubit gate from little endian to big endian notation and vice versa.

    Args:
        unitary: A numpy array of a unitary matrix in either little or big endian notation

        Returns:
        A numpy array representing the unitary matrix in big or little endian notation

    Notes:
        For additional information see https://quantumcomputing.stackexchange.com/questions/26899/how-to-convert-between-little-big-endian-unitary-forms-in-braket
    """
    qubit_count = int(np.log2(unitary.shape[0]))
    U_tensor = unitary.reshape([2] * 2 * qubit_count)
    input = list(reversed(range(qubit_count)))
    output = [i + qubit_count for i in input]
    biglittle_endian_tensor = np.einsum(U_tensor, input + output)
    return biglittle_endian_tensor.reshape([2 ** qubit_count, 2 ** qubit_count])






a = 0.25
b = 0.75

# matrix to be decomposed
A = np.array(
    [[a,  0, 0,  b],
     [0, -a, b,  0],
     [0,  b, a,  0],
     [b,  0, 0, -a]]
)

print(f"Matrix A:\n{A}\n")
print(f"Matrix A in big endian:\n{_convert_biglittle_endian_unitary(A)}\n")

LCU = qml.pauli_decompose(A)
LCU_coeffs, LCU_ops = LCU.terms()

print(f"LCU decomposition:\n {LCU}")
print(f"Coefficients:\n {LCU_coeffs}")
print(f"Unitaries:\n {LCU_ops}")

name_map_pennylane_to_cudaq = {
    "Identity": "eye",
    "PauliX": "x",
    "PauliY": "y",
    "PauliZ": "z"
}

ops_qubits = tuple((pauli.name, pauli.wires[0]) for word in LCU_ops for pauli in word)
print(ops_qubits)
ops = tuple(name_map_pennylane_to_cudaq[op[0]] for op in ops_qubits)
qubits = tuple(op[1] for op in ops_qubits)

ops = tuple(tuple(name_map_pennylane_to_cudaq[op.name] for op in word) for word in LCU_ops)
qubits = tuple(tuple(op.wires[0] for op in word) for word in LCU_ops)

print(ops, qubits)


cudaq.register_operation('eye', np.array([1., 0., 0., 1.]))
coeffs_to_angles = np.sqrt(LCU_coeffs)/np.linalg.norm(np.sqrt(LCU_coeffs))
print(f"Coefficients to angles: {coeffs_to_angles}")

# angle_state_prep = float(2*np.arccos(coeffs_to_angles[0]))
# print(f"angle_state_prep: {angle_state_prep} radians")
# @cudaq.kernel
# def lcu_kernel_le(angle_state_prep:float):
#     qvec = cudaq.qvector(3)
#     ry(angle_state_prep, qvec[0])
#     x(qvec[0])
#     eye.ctrl(qvec[0], qvec[1])
#     z.ctrl(qvec[0], qvec[2])
#     x(qvec[0])
#     x.ctrl(qvec[0], qvec[1])
#     x.ctrl(qvec[0], qvec[2])

# cudaq.set_target('nvidia', options='fp64')
# state = cudaq.get_state(lcu_kernel_le, angle_state_prep)
# print(state)
# print(np.power(state, 2))

print()

angle_state_prep = float(2*np.arccos(coeffs_to_angles[0]))
@cudaq.kernel
def lcu_kernel_be(angle_state_prep:float):
    qvec = cudaq.qvector(3)
    ry(angle_state_prep, qvec[0])
    x(qvec[0])
    eye.ctrl(qvec[0], qvec[1])
    z.ctrl(qvec[0], qvec[2])
    x(qvec[0])
    x.ctrl(qvec[0], qvec[1])
    x.ctrl(qvec[0], qvec[2])
    ry(-angle_state_prep, qvec[0])
    
    
cudaq.set_target('nvidia', options='fp64')
state = cudaq.get_state(lcu_kernel_be, angle_state_prep)
state_pow2 = np.power(state, 2)
print(state)
#print(state_pow2)
print(state.amplitudes(['000', '100']))
print(np.sum(np.power(state.amplitudes(['000', '100']), 2)))
print(np.sum(np.power(state.amplitudes(['001', '101']), 2)))
print(np.sum(np.power(state.amplitudes(['010', '110']), 2)))
print(np.sum(np.power(state.amplitudes(['011', '111']), 2)))
print()
print(np.sum(np.power(state.amplitudes(['000', '001']), 2)))
