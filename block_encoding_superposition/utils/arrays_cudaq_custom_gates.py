import numpy as np
import typing
from typing import List

def create_adder_array(int_to_add:int, num_qubits:int) -> np.ndarray:
    adder_array = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
    for i in range(2**num_qubits):
        # Convert the index to binary and add the integer
        new_index = (i + int_to_add) % (2**num_qubits)
        adder_array[new_index, i] = 1.0  # Set the amplitude for the new index
            
    return adder_array

def create_block_encoding_array(a: np.ndarray, normalize_to_1: bool) -> np.ndarray:
    '''Following qml.BlockEncode'''
    if normalize_to_1:
        norm = np.max(np.abs(np.linalg.svd(a@a.conjugate().T, compute_uv=False)))**2
        a /= norm
    
    offdiag_upper = np.eye(a.shape[0]) - a @ a.conjugate().T
    offdiag_lower = np.eye(a.shape[0]) - a.conjugate().T @ a
    
    U, S, Vh = np.linalg.svd(offdiag_upper)
    offdiag_diag = np.diag(S)
    offdiag_upper = U @ np.sqrt(offdiag_diag) @ Vh 
    U, S, Vh = np.linalg.svd(offdiag_lower)
    offdiag_diag = np.diag(S)
    offdiag_lower = U @ np.sqrt(offdiag_diag) @ Vh
    
    block_array = np.block([[a, offdiag_upper], [offdiag_upper, -a.conjugate().T]])

    return block_array