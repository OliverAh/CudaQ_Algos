print('Start imports ...')
# import os
# print('... imported os')
# import cudaq
# print('... imported cudaq')
import pennylane as qml
print('... imported pennylane')
import numpy as np
print('... imported numpy')
import sys
print('... imported sys')
import pathlib
print('... imported pathlib')
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from src import qsvt
print('... imported qsvt')
import utils
print('... imported utils')
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.joinpath('aqualib', 'src')))
import topology_optimization as to
import topology_optimization.block_encoding
print('... imported topology_optimization from aqualib')
print('Imports done.')

biglittleendianconversion = lambda A: qsvt.QSVT._convert_biglittle_endian_unitary(1,A)

def _test_big_little_endian_conversion():
    A = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]], dtype=np.complex128)
    B = np.array([[1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 1, 0, 0]], dtype=np.complex128)

    assert np.allclose(biglittleendianconversion(A), B)
    assert np.allclose(biglittleendianconversion(B), A)

    return
_test_big_little_endian_conversion()
del _test_big_little_endian_conversion



if True:
    nelx = 2 # number of elements in x-direction
    nely = 2 # number of elements in y-direction

    ndof = 2 * (nelx + 1) * (nely + 1)
    local_num_qubits = 3 #?np.log2(ndof-constrained_dofs)?
    global_num_qubits = int(np.log2((nelx*nely)*(2**local_num_qubits))) #5 number of qubits to encode global stiffness matrix of single configuration of elements

    translations = [0, 2, 6, 8] # translation (in dof) of local stiff. mat. in global stiff. mat.

    gap_size = 2 # gap size (in dof) of local stiff. mat. within global stiff. mat.
    gap_start = 4 # start point of gap (in dof), measured from first dof of element

    rmin = 1.5
    penal = 3
    Emin = 0
    Emax = 1
    
    u = list(range(nelx*nely)) #[0, 1, 2, 3] # contains element numbers of elements that are available for topology optimization
    data = list(range(u[-1]+1, u[-1]+1+global_num_qubits)) #[4, 5, 6, 7, 8] # holds |b> and versions of K
    flag = data[-1] + 1 #9 #???
    local_block = flag + 1 #10 # ancilla qubit for block encoding of K matrix
    control_flag = local_block + 1 #11 # anc. to indicate which element configuration is encoded?
    lcu_var = list(range(control_flag+1, control_flag+1+int(np.log2((nelx*nely))))) #[12, 13] # ???
    _qubit_forest = [[flag], [local_block], [control_flag], lcu_var]
    zero_condition_wires = [leaf for tree in _qubit_forest for leaf in tree] #[9, 10, 11, 12, 13] # qubits that need to evaluate to 0 during readout
    del _qubit_forest
    total_wires = zero_condition_wires[-1] + 1 #14 # +1 because first qubit has id 0

    print('Qubit registers:')
    print('    nelx:', nelx)
    print('    nely:', nely)
    print('    ndof:', ndof)
    print('    local_num_qubits:', local_num_qubits)
    print('    global_num_qubits:', global_num_qubits)
    print('    translations:', translations)
    print('    gap_size:', gap_size)
    print('    gap_start:', gap_start)
    print('    rmin:', rmin)
    print('    penal:', penal)
    print('    Emin:', Emin)
    print('    Emax:', Emax)
    print('    u:', u)
    print('    data:', data)
    print('    flag:', flag)
    print('    local_block:', local_block)
    print('    control_flag:', control_flag)
    print('    lcu_var:', lcu_var)
    print('    zero_condition_wires:', zero_condition_wires)
    print('    total_wires:', total_wires)
    print()



array_of_matrices_classical, array_of_matrices_not_rescaled_classical, _ = utils.classical_construction_of_global_matrix.construct_global_matrices_classically(
            nelx, nely, rmin, penal, Emin, Emax, ndof, []
        )
# with np.printoptions(precision=3, suppress=True, linewidth=200):
#     print('Classical matrices (rescaled):')
#     print(array_of_matrices_classical)
#     print('Classical matrices (not rescaled):')
#     print(array_of_matrices_not_rescaled_classical)

element_stiffness_matrix = utils.classical_construction_of_global_matrix.lk()
# with np.printoptions(precision=3, suppress=True, linewidth=200):
#     print('Element stiffness matrix:')
#     print(element_stiffness_matrix)


a = qml.matrix(qml.Adder(1, [0,1]))
with np.printoptions(precision=3, suppress=True, linewidth=200):
    print(a)

# @cudaq.kernel
# def block_encoding_kernel():
#     qvec_u = cudaq.qvector(len(u))
#     qvec_global_k = cudaq.qvector(global_num_qubits)
#     qbit_flag = cudaq.qubit()
#     qbit_local_block = cudaq.qubit()
#     qbit_control_flag = cudaq.qubit()
#     qvec_lcu_var = cudaq.qvector(len(lcu_var))

#     h(qvec_u)
#     h(qvec_global_k)
#     h(qbit_flag)
#     h(qbit_local_block)
#     h(qbit_control_flag)
#     h(qvec_lcu_var)

# drawing = cudaq.draw(block_encoding_kernel)
# print(drawing)
    # choose activate elements

# array_of_matrices_quantum, array_of_matrices_not_rescaled_quantum = utils.construct_quantum_matrices.construct_matrices_for_all_configurations(
#            u, data, flag, local_block, control_flag, lcu_var, zero_condition_wires, total_wires,
#            global_num_qubits, local_num_qubits, translations, gap_size, gap_start
#        )
    
# with np.printoptions(precision=3, suppress=True, linewidth=400, threshold=sys.maxsize):
#     print('Quantum matrices (rescaled):')
#     print(array_of_matrices_quantum[0].real)
#     print(array_of_matrices_quantum[1].real)
#     print('Quantum matrices (not rescaled):')
#     print(array_of_matrices_not_rescaled_quantum[0].real) 
#     print(array_of_matrices_not_rescaled_quantum[1].real)

adder_arrays = []
num_qubits_adder = 3
for i in range(5):
    adder_arrays.append(utils.arrays_cudaq_custom_gates.create_adder_array(i, num_qubits_adder))
with np.printoptions(precision=3, suppress=True, linewidth=400, threshold=sys.maxsize):
    print('Adder arrays:')
    for i, adder_array in enumerate(adder_arrays):
        print(f'Adder array {i}:')
        print(adder_array)
        print()

svals_kl = np.linalg.svd(element_stiffness_matrix, compute_uv=False)
        
block_array = utils.arrays_cudaq_custom_gates.create_block_encoding_array(element_stiffness_matrix, normalize_to_1=True)
with np.printoptions(precision=3, suppress=True, linewidth=400, threshold=sys.maxsize):
    print('Singular values of element_stiffness_matrix:', svals_kl)
    print('Block encoding array:')
    print(block_array)
print(np.linalg.svd(block_array, compute_uv=False))
print(np.linalg.svd(block_array[:8,:8], compute_uv=False))
print()

block_unitary = to.block_encoding.construct_unitary(element_stiffness_matrix)
with np.printoptions(precision=3, suppress=True, linewidth=400, threshold=sys.maxsize):
    print('Block encoding unitary:')
    print(block_unitary)
print(np.linalg.svd(block_unitary, compute_uv=False))
print(np.linalg.svd(block_unitary[:8,:8], compute_uv=False))