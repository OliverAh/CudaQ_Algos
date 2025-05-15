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
        if _construct_string_kernel_initialize_b_register is None:
            self._construct_string_kernel_initialize_b_register = self._construct_string_kernel_initialize_b_register_all_ones_sysem_size_4
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

#    def compute_pauli_decomposition_A_block_encoded(self) -> None:
#        if self.verbose > 2:
#            print('Start to compute Pauli decomposition')
#        pd = qml.pauli_decompose(self.A_block_encoded_unitary, pauli=False, hide_identity=False)
#        self.pauli_decomposition_coeffs = pd.coeffs
#        self.pauli_decomposition_paulis = []
#        for op in pd.ops:
#            self.pauli_decomposition_paulis.append(''.join(c for c in str(op) if c.isupper()))
#        if self.verbose > 2:
#            print('Finished to compute Pauli decomposition')
#        return None
#    
#    def _pauliword_to_matrix(self, pauliword:str) -> np.ndarray:
#        """
#        Convert a Pauli word to a matrix
#        """
#        a = 1
#        for i in range(len(pauliword)):
#            if pauliword[i] == 'I':
#                a = np.kron(a, np.eye(2))
#            elif pauliword[i] == 'X':
#                a = np.kron(a, np.array([[0, 1], [1, 0]]))
#            elif pauliword[i] == 'Y':
#                a = np.kron(a, np.array([[0, 1j], [-1j, 0]]))
#            elif pauliword[i] == 'Z':
#                a = np.kron(a, np.array([[1, 0], [0, -1]]))
#            else:
#                print('Invalid Pauli word')
#                sys.exit(0)
#        return a
#    
#    def _construct_string_register_operations_paulis(self) -> Tuple[str, List[str]]:
#        s = ''
#        ops_names = []
#        coeffs = self.pauli_decomposition_coeffs
#        paulis = self.pauli_decomposition_paulis
#        for i in range(len(self.pauli_decomposition_coeffs)):
#            _pauli = paulis[i]
#            a = self._pauliword_to_matrix(_pauli)
#            for qc in range(self.qvector_clock_size):
#                _coeff = coeffs[i] * t[j] / r * 2**qc
#                mexp = scipy.linalg.expm(1j * _coeff * a)
#                mexp_adj = mexp.conjugate().transpose()
#                mexp_alt = np.cos(_coeff) * np.eye(a.shape[0]) + 1j * np.sin(_coeff) * a
#                assert np.allclose(mexp, mexp_alt), 'some problem with the matrix exponential \n'+str(mexp)+'\n'+str(mexp_alt)
#                _ops_name = 'exp__'+_pauli+'_t'+str(j)+'r'+str(rr)+'c'+str(qc)
#                ops_names.append(_ops_name)
#                s += 'cudaq.register_operation(\''+_ops_name   +'\', np.array(' + np.array2string(mexp.astype(    np.complex128).flatten(),precision=16,floatmode='maxprec',formatter={'complex_kind': lambda x: f'{x:.16e}'},separator=',').replace('\n', '') + ', dtype=np.complex128))\n'
#                adj_ops_name = 'adj_'+_ops_name
#                ops_names.append(adj_ops_name)
#                s += 'cudaq.register_operation(\''+adj_ops_name+'\', np.array(' + np.array2string(mexp_adj.astype(np.complex128).flatten(),precision=16,floatmode='maxprec',formatter={'complex_kind': lambda x: f'{x:.16e}'},separator=',').replace('\n', '') + ', dtype=np.complex128))\n'
#
#        return (s, ops_names)
    
    def _construct_string_register_operation_A_block_encoded(self) -> Tuple[str, List[str]]:
        s = ''
        ops_names = []
        
        a = self.A_block_encoded_unitary
        a_adj = a.conjugate().transpose()
        num_qubits_a = self.log_system_size_block_encoded
        
        
        for qc in range(self.qvector_clock_size):
            _ops_name = 'Block_A'
            ops_names.append(_ops_name)
            s += 'cudaq.register_operation(\''+_ops_name   +'\', np.array(' + np.array2string(a.astype(    np.complex128).flatten(),precision=16,floatmode='maxprec',formatter={'complex_kind': lambda x: f'{x:.16e}'},separator=',').replace('\n', '') + ', dtype=np.complex128))\n'
            adj_ops_name = 'adj_'+_ops_name
            ops_names.append(adj_ops_name)
            s += 'cudaq.register_operation(\''+adj_ops_name+'\', np.array(' + np.array2string(a_adj.astype(np.complex128).flatten(),precision=16,floatmode='maxprec',formatter={'complex_kind': lambda x: f'{x:.16e}'},separator=',').replace('\n', '') + ', dtype=np.complex128))\n'

        return (s, ops_names)
    
    def _construct_string_register_operations_projectors(self) -> Tuple[str, List[str]]:
        s = ''
        ops_names = []
        _angles = self.angles_poly_oneoverx
        num_qubits_A = self.log_system_size_block_encoded
        for i in range(len(_angles)):
            projector = np.zeros_like(self.A_block_encoded_unitary)
            projector[:self.system_size, :self.system_size] = np.exp( 1j * _angles[i])
            projector[self.system_size:, self.system_size:] = np.exp(-1j * _angles[i])
            
            projector_adj = projector.conjugate().transpose()
            
            _ops_name = 'pi__'+'{:.2f}'.format(_angles[i])
            ops_names.append(_ops_name)
            s += 'cudaq.register_operation(\''+_ops_name   +'\', np.array(' + np.array2string(projector.astype(    np.complex128).flatten(),precision=16,floatmode='maxprec',formatter={'complex_kind': lambda x: f'{x:.16e}'},separator=',').replace('\n', '') + ', dtype=np.complex128))\n'
            adj_ops_name = 'adj_'+_ops_name
            ops_names.append(adj_ops_name)
            s += 'cudaq.register_operation(\''+adj_ops_name+'\', np.array(' + np.array2string(projector_adj.astype(np.complex128).flatten(),precision=16,floatmode='maxprec',formatter={'complex_kind': lambda x: f'{x:.16e}'},separator=',').replace('\n', '') + ', dtype=np.complex128))\n'

        return (s, ops_names)


    def _construct_string_kernel_initialize_b_register_all_ones_sysem_size_4(self) -> str:
        s = ''
        s += '    h(qvec_b[0])\n'
        s += '    x(qvec_b[1])\n'
        s += '    x.ctrl(qvec_b[0], qvec_b[1])  # CNOT gate applied with qb[0] as control\n'
        s += '    ry(np.pi, qvec_b[0])\n'
        s += '    ry(np.pi, qvec_b[1])\n'
        return s
    
    def _construct_string_kernel_qsvt(self) -> str:
        s = ''
        _angles = self.self.angles_poly_oneoverx
        qubits_applied = list(range(self.qvector_b_size))
        
        s += '    '+'h(qvec_a[0])'
        
        _ops_name = 'pi__'+'{:.2f}'.format(_angles[0])
        s += '    '+_ops_name+'.ctrl(!qvec_a[0], qvec_c'+str(qubits_applied)+ ')\n'
        for i in range(1, len(_angles)):
            _ops_name = 'pi__'+'{:.2f}'.format(_angles[i])
            s += '    '+'Block_A'+'.ctrl(!qvec_a[0], qvec_c'+str(qubits_applied)+ ')\n'
            s += '    '+_ops_name+'.ctrl(!qvec_a[0], qvec_c'+str(qubits_applied)+ ')\n'
        
        for i in range(len(_angles)-1, 0, -1):
            _ops_name = 'adj_'+'pi__'+'{:.2f}'.format(_angles[i])
            s += '    '+'adj_Block_A'+'.ctrl(!qvec_a[0], qvec_c'+str(qubits_applied)+ ')\n'
            s += '    '+_ops_name+'.ctrl(qvec_a[0], qvec_c'+str(qubits_applied)+ ')\n'
        _ops_name = 'adj_'+'pi__'+'{:.2f}'.format(_angles[-1])
        s += '    '+_ops_name+'.ctrl(!qvec_a[0], qvec_c'+str(qubits_applied)+ ')\n'
        
        s += '    '+'h(qvec_a[0])'
        return s
    

    def construct_string_qsvt_complete(self) -> None:
        """
        Construct the string of the kernel for the complete QSVT algorithm
        """
        s_qsvt_complete = ''
        s_qsvt_complete += 'import cudaq\n'
        s_qsvt_complete += 'import numpy as np\n\n'
    
        s_operations_A_block_encoded, operations_A_block_encoded_names = self._construct_string_register_operation_A_block_encoded()
        s_qsvt_complete += s_operations_A_block_encoded + '\n'

        if self.verbose > 2:
            print('Finished _construct_string_register_operation_A_block_encoded')

        s_operations_projectors, operations_projectors_names = self._construct_string_register_operations_projectors()
        s_qsvt_complete += s_operations_projectors + '\n'

        if self.verbose > 2:
            print('Finished _construct_string_register_operations_projectors')


        s_qsvt_complete += '@cudaq.kernel\n'
        s_qsvt_complete += 'def qsvt():\n'
        s_qsvt_complete += '    qbit_a = cudaq.qubit()\n'
        s_qsvt_complete += '    qvec_b = cudaq.qvector('+str(self.qvector_b_size)+')\n'
    
        s_qsvt_complete += '\n'
        s_qsvt_complete += '    ####################\n'
        s_qsvt_complete += '    # init b register\n'
        s_qsvt_complete += '    ####################\n'
        s_qsvt_complete += '\n'
        
        s_initialization_b_register = self._construct_string_kernel_initialize_b_register()
        s_qsvt_complete += s_initialization_b_register + '\n'

        if self.verbose > 2:
            print('Finished _construct_string_kernel_initialize_b_register')

    
        s_qsvt_complete += '\n'
        s_qsvt_complete += '    ####################\n'
        s_qsvt_complete += '    # apply hamiltonian simulation as part of qpe\n'
        s_qsvt_complete += '    ####################\n'
        s_qsvt_complete += '\n'
    
        s_qsvt = self._construct_string_kernel_qsvt()
        s_qsvt_complete += s_qsvt + '\n'
        #s_qsvt_complete += '    h(qvec_b)\n'
        #s_qsvt_complete += '    h(qvec_c)\n'

        if self.verbose > 2:
            print('Finished _construct_string_kernel_qsvt')

        
        if self.quantum_registers_to_measure is not None:
            s_qsvt_complete += '\n'
            s_qsvt_complete += '    ####################\n'
            s_qsvt_complete += '    # measure b register\n'
            s_qsvt_complete += '    ####################\n'
            s_qsvt_complete += '\n'
        
            for qr in self.quantum_registers_to_measure:
                s_qsvt_complete += '    mz('+qr+')\n'
        else:
            pass
        #s_qsvt_complete += '    mz(qbit_a)\n'
        #s_qsvt_complete += '    mz(qvec_c)\n'
        #s_qsvt_complete += '    mz(qvec_b)\n'
    
        self.string_kernel_qsvt_complete = s_qsvt_complete

        return None
    
    def write_and_import_kernel_qsvt_complete(self, remove_file_after_import:bool=True) -> None:
        unique_str = uuid.uuid4()
        filepath = 'tmp'
        filename = f'kernel_qsvt_complete_from_class_{unique_str}.py'
        filename = filename.replace('-', '_')
        path = pathlib.Path(filepath, filename)
        if self.verbose > 0:
            print('Wrote kernel to:', path)
        with open (path, 'w') as f:
            f.write(self.string_kernel_qsvt_complete)

        spec = importlib.util.spec_from_file_location('kernel_qsvt_complete_from_class', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules['kernel_qsvt_complete_from_class'] = module
        spec.loader.exec_module(module)
        #from tmp import kernel_qsvt_complete_from_class
        #exec(f'from tmp import {filename[:-3]} as kernel_qsvt_complete_from_class')

        if self.verbose > 2:
            print('Imported kernel as module')

        self.kernel_qsvt_complete = cudaq.PyKernelDecorator.from_json(module.qsvt.to_json())

        if self.verbose > 2:
            print('Created kernel from module')

        if remove_file_after_import:
            path.unlink()
            path_pycache = pathlib.Path(filepath, '__pycache__')
            for p in list(path_pycache.glob(filename.split('.')[0] + '*')):
                # list should be of length one
                p.unlink()

        
        return None
    
    def draw(self) -> str:
        self.circuit_string = cudaq.draw(self.kernel_qsvt_complete)
        return self.circuit_string
    
    def sample(self, **kwargs) -> cudaq.SampleResult:
        '''kwargs are passed to cudaq.sample(kernel, **kwargs)'''
        
        cudaq.set_target(self.cudaq_target, option=self.cudaq_target_option)
        if kwargs.get('shots_count', None) is None:
            self.samples_shots_count = 1000
        else:
            self.samples_shots_count = kwargs['shots_count']
        
        self.samples = cudaq.sample(self.kernel_qsvt_complete, **kwargs)
        return self.samples
    
    def sample_async(self, **kwargs) -> None:
        '''kwargs are passed to cudaq.sample(kernel, **kwargs)'''
        if self.verbose > 2:
            print('Set cudaq target')
        cudaq.set_target(self.cudaq_target, option=self.cudaq_target_option)
        if kwargs.get('shots_count', None) is None:
            self.samples_shots_count = 1000
        else:
            self.samples_shots_count = kwargs['shots_count']
        if self.verbose > 2:
            print('Start sample_async')
        self.samples = cudaq.sample_async(self.kernel_qsvt_complete, **kwargs)
        if self.verbose > 2:
            print('Finished sample_async')
        return None
    
    def get_state(self, **kwargs) -> cudaq.State:
        '''kwargs are passed to cudaq.get_state(kernel, **kwargs)'''
        cudaq.set_target(self.cudaq_target, option=self.cudaq_target_option)
        for meas in ['mz', 'mx', 'my']:
            if meas in self.string_kernel_qsvt_complete:
                raise ValueError('Measurement in kernel_qsvt_complete is not allowed, when requesting the quantum state')
        self.quantum_state = cudaq.get_state(self.kernel_qsvt_complete, **kwargs)
        return self.quantum_state
    
    def get_state_async(self, **kwargs) -> None:
        '''kwargs are passed to cudaq.get_state(kernel, **kwargs)'''
        cudaq.set_target(self.cudaq_target, option=self.cudaq_target_option)
        for meas in ['mz', 'mx', 'my']:
            if meas in self.string_kernel_qsvt_complete:
                raise ValueError('Measurement in kernel_qsvt_complete is not allowed, when requesting the quantum state')
        self.quantum_state = cudaq.get_state_async(self.kernel_qsvt_complete, **kwargs)
        return None
    
    def create_samples_dict_ordered_be_and_reduced_b_be(self) -> None:
        assert self.samples is not None, 'need to sample before ordering samples'
        #sampled_bitstrings_be = list([k for k,_ in self.samples.items()])
        if self.verbose > 0:
            print('Create samples_dict_orig')
        samples_dict_orig = {k:v for k,v in self.samples.items()}
        if self.verbose > 0:
            print('Create samples_dict_ordered_be')
        samples_dict_ordered_be = {bs: samples_dict_orig.get(bs, 0) for bs in tqdm.tqdm(self.bit_strings_big_endian_all, disable=not self.verbose > 0)}# if bs in sampled_bitstrings_be}
        self.samples_dict_ordered_be = samples_dict_ordered_be
        if self.verbose > 0:
            print('Create samples_dict_ordered_reduced_b_be')
        samples_dict_ordered_reduced_b_be = {bs: sum([val for key, val in samples_dict_ordered_be.items() if key.endswith(bs) and key[0]=='1']) for bs in tqdm.tqdm(self.bit_strings_big_endian_qvector_b, disable=not self.verbose > 0)}
        if self.verbose > 0:
            print('Finished creating samples_dict_ordered_reduced_b_be:\n', samples_dict_ordered_reduced_b_be)
        self.samples_dict_ordered_reduced_b_be = samples_dict_ordered_reduced_b_be
        return None
    
    def create_quantum_state_amplitudes_dict_ordered_be(self) -> None:
        assert self.quantum_state is not None, 'need to get_state before ordering state amplitudes'
        if self.verbose > 0:
            print('Create state_amplitudes_dict_ordered_be')
        state_amplitudes_dict_ordered_be = {tqdm.tqdm(zip(self.bit_strings_big_endian_all, self.quantum_state.amplitudes(self.bit_strings_big_endian_all)), disable=not self.verbose > 0)} 
        self.state_amplitudes_dict_ordered_be = state_amplitudes_dict_ordered_be
        return None