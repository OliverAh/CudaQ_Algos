import cudaq
import numpy as np
import scipy
import scipy.linalg
import matplotlib.pyplot as plt
import pennylane as qml

import tqdm

from typing import List, Callable, Tuple
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from src import hhl

abcd = hhl.HHL(qvector_clock_size=4)
abcd.t_hamiltonian_simulation = [np.max(abcd.eigvals)]
#abcd.t_hamiltonian_simulation = [1/(2*np.pi)]
#abcd.t_hamiltonian_simulation = [1.5*2*(np.pi**2)/np.max(abcd.eigvals) / abcd.system_size]
print('t_hamiltonian_simulation:', abcd.t_hamiltonian_simulation)
abcd.construct_string_hhl_complete()
print(abcd.pauli_decomposition_paulis)
#with open ('kernels/kernel_hhl_complete_class.py', 'w') as f:
#    f.write(abcd.string_kernel_hhl_complete)

abcd.write_and_import_kernel_hhl_complete()
shots_count = 1000000
#samples = cudaq.sample(abcd.kernel_hhl_complete, shots_count=shots_count)

samples = abcd.sample(shots_count=shots_count)
#print(samples)
print(type(samples))
abcd.create_samples_dict_ordered_be_and_reduced_b_be()

state = abcd.get_state()
abcd.create_quantum_state_amplitudes_dict_ordered_be()
print(abcd.state_amplitudes_dict_ordered_be)

#print('samples:', samples)
#samples_count = np.array(list(samples.values())) # not sufficient, because unordered
samples_count = np.array(list(abcd.samples_dict_ordered_reduced_b_be.values()))
print('samples_count:', samples_count)
print('samples_count[1]/samples_count[0]:', samples_count[1]/samples_count[0])
print('w[1]/w[0]:', abcd.classical_solution[1]/abcd.classical_solution[0])
print('max(samples_count)/min(samples_count):', np.max(samples_count)/np.min(samples_count))
print('max(w)/min(w):', np.max(abcd.classical_solution)/np.min(abcd.classical_solution))
quantum_solution = samples_count / shots_count
print('samples normalized:', quantum_solution)
print('ratio Ax / b:', [(abcd.A_hermitian@quantum_solution)[i] / abcd.b_hermitian[i] for i in range(len(abcd.b_hermitian))])
print('Ax, b:', abcd.A_hermitian@quantum_solution, abcd.b_hermitian)
scaling_factor = np.average([abcd.b_hermitian[i] / (abcd.A_hermitian@quantum_solution)[i] for i in range(len(abcd.b_hermitian))])
print('scaling_factor:', scaling_factor)
quantum_solution *= scaling_factor
print(quantum_solution)

x = range(len(abcd.b_hermitian))
plt.plot(x, abcd.classical_solution, label='classical')
plt.plot(x, quantum_solution, label='quantum')
#plt.plot(x,1/np.linalg.norm(b_herm)/shots_count*np.array(list(samples.values())), label='quantum')
plt.legend()
plt.savefig('classical_vs_quantum_sol.png')

#print('state:', state)
#num_qubits_ancilla = abcd.qvector_ancilla_size
#num_qubits_clock = abcd.qvector_clock_size
#num_qubits_b = abcd.qvector_b_size
#num_qubits = num_qubits_ancilla + num_qubits_clock + num_qubits_b
#print('num_qubits:', num_qubits)
#for i in range(2**num_qubits):
#    bit_string_big_endian = format(i, '0' + str(num_qubits) + 'b')
#    bit_string_big_endian = bit_string_big_endian[::-1]
#    print(i, bit_string_big_endian, shots_count * np.abs(state.amplitude(bit_string_big_endian))**2)

#print(state.amplitude('0000001'))

