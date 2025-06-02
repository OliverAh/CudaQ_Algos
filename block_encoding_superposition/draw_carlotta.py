import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.joinpath('aqualib', 'src')))

import topology_optimization as to
import topology_optimization.extraction
import numpy as np

global_num_qubits = 5
local_num_qubits = 3

translations = [0, 2, 6, 8]

gap_size = 2
gap_start = 4

nelx = 2
nely = 2
rmin = 1.5
penal = 3
Emin = 0
Emax = 1
ndof = 2 * (nelx + 1) * (nely + 1)

u = [0, 1, 2, 3]
data = [4, 5, 6, 7, 8]
flag = 9
local_block = 10
control_flag = 11
lcu_var = [12, 13]
zero_condition_wires = [9, 10, 11, 12, 13]
total_wires = 14

array_of_matrices_quantum, array_of_matrices_not_rescaled_quantum = to.extraction.construct_matrices_for_all_configurations(
            u, data, flag, local_block, control_flag, lcu_var, zero_condition_wires, total_wires,
            global_num_qubits, local_num_qubits, translations, gap_size, gap_start
        )

with np.printoptions(precision=3, suppress=True, linewidth=400, threshold=sys.maxsize):
    print('Quantum matrices (rescaled):')
    print(array_of_matrices_quantum[0].real)
    print(array_of_matrices_quantum[1].real)
    print('Quantum matrices (not rescaled):')
    print(array_of_matrices_not_rescaled_quantum[0].real) 
    print(array_of_matrices_not_rescaled_quantum[1].real) 