import cudaq
import numpy as np
import scipy
import scipy.linalg
import matplotlib.pyplot as plt
import pennylane as qml
import pyqsp
import pathlib
import tqdm
import uuid
import importlib.util

from typing import List, Callable, Tuple
import sys


class QSVT:
    def __init__(self,
                 system_size:int=4,
                 A:np.ndarray|None=None,
                 b:np.ndarray|None=None,
                 angles_poly_oneoverx:List[float]|None=None,
                 init_Ab:Callable[[int],Tuple[np.ndarray,np.ndarray,float]]|None=None,
                 _construct_string_kernel_initialize_b_register:Callable[[],str]|None=None,
                 cudaq_target:str|None='nvidia',
                 cudaq_target_option:str|None='fp64',
                 compute_classical_solution_on_init:bool|None=True,
                 compute_eigvals_on_init:bool|None=True,
                 quantum_registers_to_measure:List[str]|None=None,
                 verbose:int=99
                 ):
        self.system_size = system_size
        self.A = A
        self.b = b
        self.angles_poly_oneoverx = angles_poly_oneoverx
        self.init_Ab = init_Ab
        self.cudaq_target = cudaq_target
        self.cudaq_target_option = cudaq_target_option
        self.compute_classical_solution_on_init = compute_classical_solution_on_init
        self.compute_eigvals_on_init = compute_eigvals_on_init
        self.quantum_registers_to_measure = quantum_registers_to_measure
        self.verbose = verbose


        self.A_condition_number = None
        self.A_block_encoded_unitary = None
        self.b_block_encoded_normalized = None
        self.log_system_size_block_encoded = None
        self.qvector_b_size = None
        self.classical_solution = None
        self.eigvals = None
        self.circuit_pennylane = None
        self.circuit_string = None
        self.samples = None
        self.quantum_state = None


        if self.init_Ab is None:
            self.init_Ab = self._init_Ab_poisson_first_order_FD
        if self.A is None and self.b is None:
            _A, _b, _alpha = self.init_Ab()
            self.A = _A
            self.b = _b
            self.alpha = _alpha
        
        #self.hermitianize_system()
        self.BlockEncode_A_unitary()
        self.determine_qvector_sizes()

        if compute_classical_solution_on_init:
            self.compute_classical_solution()
        if compute_eigvals_on_init:
            self.compute_eigvals()

        
    def _init_Ab_poisson_first_order_FD(self) -> Tuple[np.ndarray,np.ndarray,float]:
        """
        Initialize A and b for the Poisson equation with first order FDE
        """
        size = self.system_size
        # Define alpha as the value in front of b vector, i.e. Ax=alpha*b
        alpha = 12000*((5/7)**4)/(2e09 * 3.375e-4)
        A = np.zeros((size,size))

        tmp = [-1, 2, -1]
        A[0,0:2] = [2, -1]
        A[-1,-2:] = [-1, 2]
        for i in range(1,size-1):
            A[i,i-1:i+2] = tmp
        b = np.ones((size,1))

        return (A, b, alpha)
    

    def _compute_eigendecomposition(self, a:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the eigendecomposition of matrix
        """
        eigs, v = scipy.linalg.eigh(a)
        # Check if the decomposition was successfull
        if not np.allclose(a, v @ np.diag(eigs) @ v.T):
            print('Matrix is not hermitian')
            sys.exit(0)
        return (eigs, v)

    def _compute_singularvaluedecomposition(self, a:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the eigendecomposition of matrix
        """
        U, s, Vh = scipy.linalg.svd(a, lapack_driver='gesvd')
        # Check if the decomposition was successfull
        if not np.allclose(a, U @ np.diag(s) @ Vh):
            print('SVD failed')
            sys.exit(0)
        if not np.allclose(U, Vh):
            print('SVD was successfull, but Matrix is not hermitian.')
            print(U @ Vh)
        return (U, s, Vh)

    # def compute_eigendecomposition_A(self) -> Tuple[np.ndarray, np.ndarray]:
    #     eigs, v = self._compute_eigendecomposition(self.A)
    #     np.allclose(self.A, v @ np.diag(eigs) @ v.T)

    #     return (eigs, v)
    
    def BlockEncode_A_unitary(self) -> None:
        """
        Block encode the system
        """
        if self.verbose > 0:
            print('Block encoding the system')
        # Block encoding the system
        A = self.A
        A_unitary = np.zeros_like(A)
        A_unitary = A
        _eigs, _v = self._compute_eigendecomposition(A)
        print('Condition number from eigs:', np.max(np.abs(_eigs)) / np.min(np.abs(_eigs)))
        #_U, _s, _Vh = self._compute_singularvaluedecomposition(A)
        #self.A_inverse = _U @ np.diag(1/_s) @ _Vh
        self.A_inverse = _v @ np.diag(1/_eigs) @ _v.T
        #A_unitary_scale = np.linalg.norm(A_unitary @ A_unitary, ord=2) # pennylane scaling, suboptimal because largest eigenvalue > 1, therefore smallest eigv < than necessary
        A_unitary_scale = np.max(np.abs(_eigs))
        #A_unitary /= A_unitary_scale
        A_unitary /= np.max(np.abs(_eigs))
        self.A_unitary = A_unitary
        #_U, _s, _Vh = self._compute_singularvaluedecomposition(A_unitary)
        _eigs, _v = self._compute_eigendecomposition(A_unitary)
        #self.A_unitary_inverse = _U @ np.diag(1/_s) @ _Vh
        self.A_unitary_inverse = _v @ np.diag(1/_eigs) @ _v.T
        if not np.allclose(self.A_unitary_inverse, scipy.linalg.inv(A_unitary)):
            print('Block encoding failed, inverse of A_unitary deviates')
            print(self.A_unitary_inverse)
            print(scipy.linalg.inv(A_unitary))
            sys.exit(0)
        print('A_unitary_scale:', A_unitary_scale)
        A_off_diag = np.zeros(A.shape)
        A_off_diag = np.eye(A.shape[0]) - A_unitary @ A_unitary
        #_eigs, _v = self._compute_eigendecomposition(A_off_diag)
        #_U, _s, _Vh = self._compute_singularvaluedecomposition(A_off_diag)
        _eigs, _v = self._compute_eigendecomposition(A_off_diag)
        # with np.printoptions(precision=2, linewidth=200):
            # print('SVD of A_off_diag:', _s)
            # print(_U)
            # print(_Vh)
            # print(_U @ _Vh)
        #if not np.allclose(A_off_diag, _U @ np.diag(_s) @ _Vh):
        if not np.allclose(A_off_diag, _v @ np.diag(_eigs) @ _v.T):
            #print('Block encoding failed, svd of A_off_diag deviates')
            print('Block encoding failed, eigendecomposition of A_off_diag deviates')
            #print(A_off_diag - _U @ np.diag(_s) @ _Vh)
            print(A_off_diag - _v @ np.diag(_eigs) @ _v.T)
            sys.exit(0)
        #_A_off_diag = (_U @ np.diag(np.sqrt(_s)) @ _Vh)
        _A_off_diag = _v @ np.diag(np.emath.sqrt(_eigs)) @ _v.T
        if not np.allclose(A_off_diag, _A_off_diag @ _A_off_diag):
            #print('Block encoding failed, svd of sqrt(I-AA) deviates')
            print('Block encoding failed, eigendecomposition of sqrt(I-AA) deviates')
            print(A_off_diag - _A_off_diag @ _A_off_diag)
            sys.exit(0)
        else:
            if self.verbose > 0:
                print('Block encoding successfull')
        A_off_diag = _A_off_diag
        # Create a block encoding of the system
        A_block_encoded = np.zeros((2*A.shape[0], 2*A.shape[1]), dtype=A_off_diag.dtype)
        A_block_encoded[:A.shape[0], :A.shape[1]] = A_unitary
        A_block_encoded[A.shape[0]:, :A.shape[1]] = A_off_diag
        A_block_encoded[:A.shape[0], A.shape[1]:] = A_off_diag
        A_block_encoded[A.shape[0]:, A.shape[1]:] = -A_unitary

        if not np.allclose(np.eye(A_block_encoded.shape[0]), A_block_encoded @ A_block_encoded):
            print('Block encoding failed, not unitary')
            sys.exit(0)

        self.A_block_encoded_unitary = A_block_encoded
        self.A_block_encoded_unitary_scale = A_unitary_scale

        self.b_block_encoded_normalized = np.zeros((2*self.b.shape[0], 1), dtype=self.b.dtype)
        self.b_block_encoded_normalized[:self.b.shape[0]] = self.b
        self.b_block_encoded_normalized /= np.linalg.norm(self.b_block_encoded_normalized)

        return None

    def determine_qvector_sizes(self) -> None:
        self.log_system_size_block_encoded = int(np.log2(self.A_block_encoded_unitary.shape[0]))
        self.qvector_ancilla_size = 1
        self.qvector_b_size = self.log_system_size_block_encoded
        
        num_qubits = self.qvector_ancilla_size + self.qvector_b_size
        self.num_qubits = num_qubits
        
        self.bit_strings_big_endian_all = [format(i, '0' + str(num_qubits) + 'b')[::-1] for i in range(2**num_qubits)]
        self.bit_strings_big_endian_qvector_b = [format(i, '0' + str(self.qvector_b_size) + 'b')[::-1] for i in range(2**self.qvector_b_size)]

    def compute_condition_number(self, a:np.ndarray|None=None) -> float:
        """
        Compute the condition number of matrix
        """
        if a is None:
            a = self.A
        self.A_condition_number = np.linalg.cond(a, p=2)
        if self.verbose > 0:
            print('Condition number:', self.A_condition_number)
        return self.A_condition_number

    def compute_classical_solution(self) -> None:
        self.classical_solution = np.linalg.solve(self.A, self.b)
        return None

    def compute_eigvals(self) -> None:
        self.eigvals = np.linalg.eigvals(self.A)
        self.eigvecs = np.linalg.eig(self.A)[1]
        if self.verbose > 0:
            print('Eigenvalues:\n', self.eigvals)
            print('Eigenvectors:\n', self.eigvecs)
        return None

    def construct_qsvt_circuit_pennylane(self) -> None:
        angles = self.angles_poly_oneoverx
        
        def qsvt(self, angles):
            wires=range(1, self.log_system_size_block_encoded+1)
            qml.PCPhase(angles[0], dim=self.system_size, wires=wires)
            for i in range(1,len(angles)):
                qml.BlockEncode(self.A, wires=wires)
                qml.PCPhase(angles[i], dim=self.system_size, wires=wires)
    
        @qml.qnode(qml.device("default.qubit", wires=range(self.num_qubits)))
        def qsvt_run():
            qml.StatePrep(self.b.T/np.linalg.norm(self.b), range(self.log_system_size_block_encoded+1 - int(np.log2(self.system_size)), self.log_system_size_block_encoded+1))
        #    qml.Identity(wires=[0,1,2,3])#0is ancilla qubit
            qml.Hadamard(wires=[0])
            qml.ctrl(qsvt, control=(0,), control_values=(0,))(self, angles)
            qml.ctrl(qml.adjoint(qsvt), control=(0,), control_values=(1,))(self, angles)

            qml.Hadamard(wires=[0])

            return qml.state()
        
        self.circuit_pennylane = qsvt_run

