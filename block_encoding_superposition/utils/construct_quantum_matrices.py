import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.parent.joinpath('aqualib', 'src')))

import topology_optimization as to
import topology_optimization.extraction
import numpy as np
import pennylane as qml


def extract_data_amplitude_vector(full_state, data_wires, zero_condition_wires, total_wires):
    """
    Gibt einen Vektor der Länge 2^len(data_wires) zurück,
    der die aufsummierten Amplituden für jeden Datenzustand enthält,
    wenn alle zero_condition_wires == 0 sind.
    """
    data_dim = 2 ** len(data_wires)
    result = np.zeros(data_dim, dtype=complex)

    for i in range(2 ** total_wires):
        bits = format(i, f"0{total_wires}b")

        # Nur wenn alle anderen Register == 0
        if all(bits[w] == '0' for w in zero_condition_wires):
            data_bits = ''.join(bits[w] for w in data_wires)
            data_index = int(data_bits, 2)
            result[data_index] += full_state[i]

    return result


def truncate_small_values(vector, threshold=1e-10):
    # truncate small values
    real_part = np.copy(np.real(vector))     
    real_part[np.abs(real_part) < threshold] = 0
    
    # Reconstruct the array with filtered real parts
    return real_part


def construct_matrices_for_all_configurations(u, data,flag, local_block, control_flag, lcu_var, zero_condition_wires, total_wires, global_number_qubits, local_number_qubits, translations, gap_size,gap_start, Emin_divided_by_Emax= None, with_Emin= False, ancilla_delete_fixed_DOFs= None, fixed_DOFs= None):
    array_of_matrices = {}
    array_of_matrices_not_rescaled = {}
    for u_configuration in  range(2**len(translations)):
        u_configuration_array = [int(b) for b in format(u_configuration, f"0{len(translations)}b")]


        encoded_matrix_size = 2 ** global_number_qubits
        A_not_rescaled = np.zeros((encoded_matrix_size, encoded_matrix_size), dtype=complex)
        A = np.zeros((encoded_matrix_size, encoded_matrix_size), dtype=complex)

        for index_data in range(2**global_number_qubits):

            binary_array_data = [int(b) for b in format(index_data, f"0{global_number_qubits}b")]
           
            circuit = to.circuits.construct_qnode(u, data, flag, local_block, control_flag, lcu_var, global_number_qubits, local_number_qubits, translations, gap_size, gap_start, Emin_divided_by_Emax, with_Emin, ancilla_delete_fixed_DOFs, fixed_DOFs)
            print(to.circuits.qml.draw(circuit, show_all_wires=True)(binary_array_data, u_configuration_array))
            
            state= circuit(binary_array_data, u_configuration_array)
            local_stiffness= to.block_encoding.construct_local_stiffness()

            data_vector = extract_data_amplitude_vector(state, data, zero_condition_wires, total_wires)
            truncated_vector=truncate_small_values(data_vector,1e-6)
            rescaled_vector= truncated_vector*(np.max(np.abs(np.linalg.svd(local_stiffness)[1])))*2*len(translations)

            A_not_rescaled[:,index_data]=data_vector
            A[:, index_data] = rescaled_vector
        
        array_of_matrices[u_configuration]=A
        array_of_matrices_not_rescaled[u_configuration]=A_not_rescaled
    
    return array_of_matrices, array_of_matrices_not_rescaled

