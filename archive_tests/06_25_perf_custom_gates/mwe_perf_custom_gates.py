import cudaq
import numpy as np
import time

cudaq.set_target('nvidia', option='fp64')

###
# Define basic options and compute matrix for custom gate
###
circuit_depth = 10
num_qubits = 10 # first might be used as control qubit

array_hadamard = np.array([[1., 1.], [1., -1.]]) / np.sqrt(2)
array_custom_operation = array_hadamard.copy()

for i in range(num_qubits-1-1):
    array_custom_operation = np.kron(array_hadamard, array_custom_operation)
assert array_custom_operation.shape == (2**(num_qubits-1), 2**(num_qubits-1))

cudaq.register_operation('custom_multi_hadamard', array_custom_operation.flatten())

###
# Define kernels
#    1. Use native gates
#    2. Use custom gates
###

@cudaq.kernel
def kernel_native_gates():
    qvec = cudaq.qvector(num_qubits)
    for _ in range(circuit_depth):
        for j in range(1,num_qubits):
            h(qvec[j])
            #h.ctrl(qvec[0], qvec[j])

@cudaq.kernel
def kernel_custom_gates():
    qvec = cudaq.qvector(num_qubits)
    for _ in range(circuit_depth):
        custom_multi_hadamard(qvec[1], qvec[2], qvec[3], qvec[4], qvec[5], qvec[6], qvec[7], qvec[8], qvec[9])
        #custom_multi_hadamard.ctrl(qvec[0], qvec[1], qvec[2], qvec[3], qvec[4], qvec[5], qvec[6], qvec[7], qvec[8], qvec[9])

###
# Run kernels and measure execution time
###

print('Start sampling native gates')
tic_native = time.time()
state_native = cudaq.get_state(kernel_native_gates)
toc_native = time.time()
print('... finished sampling native gates')

print('Start sampling custom gates')
tic_custom = time.time()
state_custom= cudaq.get_state(kernel_custom_gates)
toc_custom = time.time()
print('... finished sampling custom gates', '\n')

assert np.allclose(state_native, state_custom)

print(f"Native gates time: {toc_native - tic_native:.2f} seconds")
print(f"Custom gates time: {toc_custom - tic_custom:.2f} seconds")