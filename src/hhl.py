import cudaq
import numpy as np
import scipy
import scipy.linalg
import matplotlib.pyplot as plt
import pennylane as qml
import pathlib
import tqdm
import uuid
import importlib.util

from typing import List, Callable, Tuple
import sys


class HHL:
    def __init__(self,
                 system_size:int=4,
                 A:np.ndarray|None=None,
                 b:np.ndarray|None=None,
                 init_Ab:Callable[[int],Tuple[np.ndarray,np.ndarray,float]]|None=None,
                 _construct_string_kernel_initialize_b_register:Callable[[],str]|None=None,
                 bit_accuracy:int|None=None,
                 error_threshold:float|None=None,
                 cudaq_target:str|None='nvidia',
                 cudaq_target_option:str|None='fp64',
                 t_hamiltonian_simulation:List[float]|None=[1.0],
                 r_hamiltonian_simulation:int|None=5,
                 qvector_clock_size:int|None=None,
                 compute_classical_solution_on_init:bool|None=True,
                 compute_eigvals_on_init:bool|None=True,
                 compute_pauli_decomposition_on_init:bool|None=True,
                 quantum_registers_to_measure:List[str]|None=None,
                 verbose:int=99
                 ):
        self.system_size = system_size
        self.A = A
        self.b = b
        self.init_Ab = init_Ab
        self.bit_accuracy = bit_accuracy
        self.error_threshold = error_threshold
        if verbose > 0 and self.error_threshold is not None:
            raise Warning('error_threshold not None, but not implemented yet')
        self.cudaq_target = cudaq_target
        self.cudaq_target_option = cudaq_target_option
        self.t_hamiltonian_simulation = t_hamiltonian_simulation
        self.r_hamiltonian_simulation = r_hamiltonian_simulation
        self.qvector_clock_size = qvector_clock_size
        self.compute_classical_solution_on_init = compute_classical_solution_on_init
        self.compute_eigvals_on_init = compute_eigvals_on_init
        self.compute_pauli_decomposition_on_init = compute_pauli_decomposition_on_init
        self.quantum_registers_to_measure = quantum_registers_to_measure
        self.verbose = verbose

        self.A_hermitian = None
        self.b_hermitian = None
        self.log_system_size = None
        self.qvector_ancilla_size = None
        self.qvector_b_size = None
        self.qvector_clock_size = qvector_clock_size
        self.classical_solution = None
        self.eigvals = None
        self.pauli_decomposition_coeffs = None
        self.pauli_decomposition_paulis = None
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
        
        self.hermitianize_system()
        self.determine_qvector_sizes()

        if compute_classical_solution_on_init:
            self.compute_classical_solution()
        if compute_eigvals_on_init:
            self.compute_eigvals()
        if compute_pauli_decomposition_on_init:
            self.compute_pauli_decomposition()

        
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
        b = np.ones(size)

        return (A, b, alpha)
    
    def _hermitianize(self, mat:np.matrix) -> np.ndarray:
        herm = np.zeros((2*mat.shape[0], 2*mat.shape[1]))
        mat_conj = mat.getH()
        for i in range(herm.shape[0]):
            for j in range(herm.shape[0]):
                if i < mat.shape[0] and j >= mat.shape[0]:
                    herm[i,j] = mat[i,j-mat.shape[0]]
                elif i >= mat.shape[0] and j < mat.shape[0]:
                    herm[i,j] = mat_conj[i-mat.shape[0],j]
                else:
                    pass
        return herm
    
    def hermitianize_system(self) -> None:
        A = self.A
        b = self.b
        if A.shape[0] != A.shape[1]:
            print('Not a Square Matrix')
            sys.exit(0)

        if int(np.log2(A.shape[0])) == float(np.log2(A.shape[0])):
            bs = int(np.log2(A.shape[0]))
            if self.verbose > 0:
                print('Matrix size is a power of 2. Base:', bs)
        else:
            print('Matrix size is not a power of 2')
            sys.exit(0)

        A_new = np.asmatrix(A)
        b_new = b
        if not scipy.linalg.ishermitian(A_new):
            if self.verbose > 0:
                print('Matrix A is not hermitian, system will be Hermitianized as [[A,0],[0,A^dagger]]')
                print('Matrix A:\n', A)
            # Hermitianize the input matrix
            A_hermitian = self._hermitianize(A_new)
            # Adjust the right side vector with additional zeros
            b_hermitian = np.zeros(2*b_new.shape[0])
            b_hermitian[:b_new.shape[0]] = b_new
            # Change hermitian matrix data type from matrix to array
            A_hermitian = np.asarray(A_hermitian)
        else:
            if self.verbose > 0:
                print('Matrix A is hermitian, system is not changed')
            # Do nothing, the matrix is hermitian
            A_hermitian = np.asarray(A_new)
            b_hermitian = b_new

        self.A_hermitian = A_hermitian
        self.b_hermitian = b_hermitian

        return None

    def determine_qvector_sizes(self) -> None:
        self.log_system_size = int(np.log2(self.system_size))
        self.qvector_ancilla_size = 1
        self.qvector_b_size = self.log_system_size
        if self.qvector_clock_size is None:
            self.qvector_clock_size = self.system_size

        num_qubits = self.qvector_ancilla_size + self.qvector_b_size + self.qvector_clock_size
        self.bit_strings_big_endian_all = [format(i, '0' + str(num_qubits) + 'b')[::-1] for i in range(2**num_qubits)]
        self.bit_strings_big_endian_qvector_b = [format(i, '0' + str(self.qvector_b_size) + 'b')[::-1] for i in range(2**self.qvector_b_size)]

    def compute_classical_solution(self) -> None:
        self.classical_solution = np.linalg.solve(self.A_hermitian, self.b_hermitian)
        return None

    def compute_eigvals(self) -> None:
        self.eigvals = np.linalg.eigvals(self.A_hermitian)
        self.eigvecs = np.linalg.eig(self.A_hermitian)[1]
        if self.verbose > 0:
            print('Eigenvalues:\n', self.eigvals)
            print('Eigenvectors:\n', self.eigvecs)
        return None

    def compute_pauli_decomposition(self) -> None:
        if self.verbose > 2:
            print('Start to compute Pauli decomposition')
        pd = qml.pauli_decompose(self.A_hermitian, pauli=False, hide_identity=False)
        self.pauli_decomposition_coeffs = pd.coeffs
        self.pauli_decomposition_paulis = []
        for op in pd.ops:
            self.pauli_decomposition_paulis.append(''.join(c for c in str(op) if c.isupper()))
        if self.verbose > 2:
            print('Finished to compute Pauli decomposition')
        return None
    
    def _pauliword_to_matrix(self, pauliword:str) -> np.ndarray:
        """
        Convert a Pauli word to a matrix
        """
        a = 1
        for i in range(len(pauliword)):
            if pauliword[i] == 'I':
                a = np.kron(a, np.eye(2))
            elif pauliword[i] == 'X':
                a = np.kron(a, np.array([[0, 1], [1, 0]]))
            elif pauliword[i] == 'Y':
                a = np.kron(a, np.array([[0, 1j], [-1j, 0]]))
            elif pauliword[i] == 'Z':
                a = np.kron(a, np.array([[1, 0], [0, -1]]))
            else:
                print('Invalid Pauli word')
                sys.exit(0)
        return a
    
    def _construct_string_register_operations_paulis(self) -> Tuple[str, List[str]]:
        s = ''
        ops_names = []
        t = self.t_hamiltonian_simulation
        r = self.r_hamiltonian_simulation
        coeffs = self.pauli_decomposition_coeffs
        paulis = self.pauli_decomposition_paulis
        for j in range(len(self.t_hamiltonian_simulation)):
            for rr in range(self.r_hamiltonian_simulation):
                for i in range(len(self.pauli_decomposition_coeffs)):
                    _pauli = paulis[i]
                    a = self._pauliword_to_matrix(_pauli)
                    for qc in range(self.qvector_clock_size):
                        _coeff = coeffs[i] * t[j] / r * 2**qc
                        mexp = scipy.linalg.expm(1j * _coeff * a)
                        mexp_adj = mexp.conjugate().transpose()
                        mexp_alt = np.cos(_coeff) * np.eye(a.shape[0]) + 1j * np.sin(_coeff) * a
                        assert np.allclose(mexp, mexp_alt), 'some problem with the matrix exponential \n'+str(mexp)+'\n'+str(mexp_alt)
                        _ops_name = 'exp__'+_pauli+'_t'+str(j)+'r'+str(rr)+'c'+str(qc)
                        ops_names.append(_ops_name)
                        s += 'cudaq.register_operation(\''+_ops_name   +'\', np.array(' + np.array2string(mexp.astype(    np.complex128).flatten(),precision=16,floatmode='maxprec',formatter={'complex_kind': lambda x: f'{x:.16e}'},separator=',').replace('\n', '') + ', dtype=np.complex128))\n'
                        adj_ops_name = 'adj_'+_ops_name
                        ops_names.append(adj_ops_name)
                        s += 'cudaq.register_operation(\''+adj_ops_name+'\', np.array(' + np.array2string(mexp_adj.astype(np.complex128).flatten(),precision=16,floatmode='maxprec',formatter={'complex_kind': lambda x: f'{x:.16e}'},separator=',').replace('\n', '') + ', dtype=np.complex128))\n'

        return (s, ops_names)
    
    def _construct_string_kernel_initialize_b_register_all_ones_sysem_size_4(self) -> str:
        s = ''
        s += '    h(qvec_b[0])\n'
        s += '    x(qvec_b[1])\n'
        s += '    x.ctrl(qvec_b[0], qvec_b[1])  # CNOT gate applied with qb[0] as control\n'
        s += '    ry(np.pi, qvec_b[0])\n'
        s += '    ry(np.pi, qvec_b[1])\n'
        return s
    
    def _construct_string_kernel_hamiltonian_simulation_timesteps_paulis(self) -> str:
        s = ''
        for j in range(len(self.t_hamiltonian_simulation)):
            for rr in range(self.r_hamiltonian_simulation):
                for _pauli in self.pauli_decomposition_paulis:
                    for qc in range(self.qvector_clock_size-1,-1,-1):
                        _ops_name = 'exp__'+_pauli+'_t'+str(j)+'r'+str(rr)+'c'+str(qc)
                        s += '    '+_ops_name+'.ctrl(qvec_c['+str(qc)+'],' + ','.join([f'qvec_b[{str(i)}]' for i in range(self.qvector_b_size)]) + ')\n'
        return s

    def _construct_string_kernel_hamiltonian_simulation_timesteps_paulis_dagger(self) -> str:
        s = ''
        for j in range(len(self.t_hamiltonian_simulation)-1,-1,-1):
            for rr in range(self.r_hamiltonian_simulation-1,-1,-1):
                for _pauli in self.pauli_decomposition_paulis:
                    for qc in range(self.qvector_clock_size):
                        _ops_name = 'exp__'+_pauli+'_t'+str(j)+'r'+str(rr)+'c'+str(qc)
                        _ops_name = 'adj_'+_ops_name
                        s += '    '+_ops_name+'.ctrl(qvec_c['+str(qc)+'],' + ','.join([f'qvec_b[{str(i)}]' for i in range(self.qvector_b_size)]) + ')\n'
        return s

    def construct_string_kernel_qft(self) -> str:
        s = ''
        for i in range(self.qvector_clock_size):
            s += '    h(qvec_c['+str(i)+'])\n'
            for j in range(i + 1, self.qvector_clock_size):
                #angle = (2 * np.pi) / (2**(j - i + 1)) # why +1?
                angle = (2 * np.pi) / (2**(j - i))
                s += f'    cr1({angle}, [qvec_c[{j}]], qvec_c[{i}])\n'
        for i in range(self.qvector_clock_size):
            a = (self.qvector_clock_size - 1) - i
            b = i
            if not (a == b) and not (a < b):
                s += f'    swap(qvec_c[{a}], qvec_c[{b}])\n'
        return s
    
    def _construct_string_kernel_qft_dagger(self) -> str:
        s = ''
        swaps = []
        for i in range(self.qvector_clock_size):
            a = (self.qvector_clock_size - 1) - i
            b = i
            if not (a == b) and not (a < b):
                swaps.append((a,b))
        if not len(swaps) == 0:
            swaps.reverse()
            for a,b in swaps:
                s += f'    swap(qvec_c[{a}], qvec_c[{b}])\n'
        for i in range(self.qvector_clock_size-1,-1,-1):
            for j in range(self.qvector_clock_size-1, i, -1):
                angle = -(2 * np.pi) / (2**(j - i + 1))
                s += f'    cr1({angle}, [qvec_c[{j}]], qvec_c[{i}])\n'
            s += '    h(qvec_c['+str(i)+'])\n'
        return s

    def _construct_string_kernel_ancilla_rotation(self) -> str:
        s = ''
        for i in range(1,self.qvector_clock_size):
            angle = 2*np.asin(1/(2**i))
            s += f'    ry.ctrl({angle},qvec_c['+str(i)+'],qbit_a)\n'
        return s

    def construct_string_hhl_complete(self) -> None:
        """
        Construct the string of the kernel for the complete HHL algorithm
        """
        s_hhl_complete = ''
        s_hhl_complete += 'import cudaq\n'
        s_hhl_complete += 'import numpy as np\n\n'
    
        s_operations_paulis, operations_paulis_names = self._construct_string_register_operations_paulis()
        s_hhl_complete += s_operations_paulis + '\n'

        if self.verbose > 2:
            print('Finished _construct_string_register_operations_paulis')

        s_hhl_complete += '@cudaq.kernel\n'
        s_hhl_complete += 'def hhl():\n'
        s_hhl_complete += '    qbit_a = cudaq.qubit()\n'
        s_hhl_complete += '    qvec_c = cudaq.qvector('+str(self.qvector_clock_size)+')\n'
        s_hhl_complete += '    qvec_b = cudaq.qvector('+str(self.qvector_b_size)+')\n'
    
        s_hhl_complete += '\n'
        s_hhl_complete += '    ####################\n'
        s_hhl_complete += '    # init b register\n'
        s_hhl_complete += '    ####################\n'
        s_hhl_complete += '\n'
        
        s_initialization_b_register = self._construct_string_kernel_initialize_b_register()
        s_hhl_complete += s_initialization_b_register + '\n'

        if self.verbose > 2:
            print('Finished _construct_string_kernel_initialize_b_register')

        s_hhl_complete += '\n'
        s_hhl_complete += '    ####################\n'
        s_hhl_complete += '    # init c register\n'
        s_hhl_complete += '    ####################\n'
        s_hhl_complete += '\n'
    
        s_hhl_complete += '    h(qvec_c)\n'
    
        s_hhl_complete += '\n'
        s_hhl_complete += '    ####################\n'
        s_hhl_complete += '    # apply hamiltonian simulation as part of qpe\n'
        s_hhl_complete += '    ####################\n'
        s_hhl_complete += '\n'
    
        s_hamiltonian_simulation_time_step_pauli = self._construct_string_kernel_hamiltonian_simulation_timesteps_paulis()
        s_hhl_complete += s_hamiltonian_simulation_time_step_pauli + '\n'
        #s_hhl_complete += '    h(qvec_b)\n'
        #s_hhl_complete += '    h(qvec_c)\n'

        if self.verbose > 2:
            print('Finished _construct_string_kernel_hamiltonian_simulation_timesteps_paulis')

        s_hhl_complete += '\n'
        s_hhl_complete += '    ####################\n'
        s_hhl_complete += '    # apply qft as part of qpe\n'
        s_hhl_complete += '    ####################\n'
        s_hhl_complete += '\n'
    
        s_qft = self.construct_string_kernel_qft()
        s_hhl_complete += s_qft + '\n'
    
        if self.verbose > 2:
            print('Finished construct_string_kernel_qft')
    
        s_hhl_complete += '\n'
        s_hhl_complete += '    ####################\n'
        s_hhl_complete += '    # apply controlled ancilla rotation\n'
        s_hhl_complete += '    ####################\n'
        s_hhl_complete += '\n'
        
        s_ancilla_rotation = self._construct_string_kernel_ancilla_rotation()
        s_hhl_complete += s_ancilla_rotation + '\n'
    
        if self.verbose > 2:
            print('Finished _construct_string_kernel_ancilla_rotation')
    
    
        s_hhl_complete += '\n'
        s_hhl_complete += '    ####################\n'
        s_hhl_complete += '    # apply qft_dagger as part of qpe_dagger\n'
        s_hhl_complete += '    ####################\n'
        s_hhl_complete += '\n'
    
        s_qft_dagger = self._construct_string_kernel_qft_dagger()
        s_hhl_complete += s_qft_dagger + '\n'
    
        if self.verbose > 2:
            print('Finished _construct_string_kernel_qft_dagger')

        s_hhl_complete += '\n'
        s_hhl_complete += '    ####################\n'
        s_hhl_complete += '    # apply hamiltonian simulation dagger as part of qpe_dagger\n'
        s_hhl_complete += '    ####################\n'
        s_hhl_complete += '\n'
    
        s_hamiltonian_simulation_time_step_pauli_dagger = self._construct_string_kernel_hamiltonian_simulation_timesteps_paulis_dagger()
        s_hhl_complete += s_hamiltonian_simulation_time_step_pauli_dagger + '\n'
        
        if self.verbose > 2:
            print('Finished _construct_string_kernel_hamiltonian_simulation_timesteps_paulis_dagger')

    
        s_hhl_complete += '\n'
        s_hhl_complete += '    ####################\n'
        s_hhl_complete += '    # init_dagger c register\n'
        s_hhl_complete += '    ####################\n'
        s_hhl_complete += '\n'
    
        s_hhl_complete += '    h(qvec_c)\n'
    
        
        if self.quantum_registers_to_measure is not None:
            s_hhl_complete += '\n'
            s_hhl_complete += '    ####################\n'
            s_hhl_complete += '    # measure b register\n'
            s_hhl_complete += '    ####################\n'
            s_hhl_complete += '\n'
        
            for qr in self.quantum_registers_to_measure:
                s_hhl_complete += '    mz('+qr+')\n'
        else:
            pass
        #s_hhl_complete += '    mz(qbit_a)\n'
        #s_hhl_complete += '    mz(qvec_c)\n'
        #s_hhl_complete += '    mz(qvec_b)\n'
    
        self.string_kernel_hhl_complete = s_hhl_complete

        return None
    
    def write_and_import_kernel_hhl_complete(self, remove_file_after_import:bool=True) -> None:
        unique_str = uuid.uuid4()
        filepath = 'tmp'
        filename = f'kernel_hhl_complete_from_class_{unique_str}.py'
        filename = filename.replace('-', '_')
        path = pathlib.Path(filepath, filename)
        if self.verbose > 0:
            print('Wrote kernel to:', path)
        with open (path, 'w') as f:
            f.write(self.string_kernel_hhl_complete)

        spec = importlib.util.spec_from_file_location('kernel_hhl_complete_from_class', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules['kernel_hhl_complete_from_class'] = module
        spec.loader.exec_module(module)
        #from tmp import kernel_hhl_complete_from_class
        #exec(f'from tmp import {filename[:-3]} as kernel_hhl_complete_from_class')

        if self.verbose > 2:
            print('Imported kernel as module')

        self.kernel_hhl_complete = cudaq.PyKernelDecorator.from_json(module.hhl.to_json())

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
        self.circuit_string = cudaq.draw(self.kernel_hhl_complete)
        return self.circuit_string
    
    def sample(self, **kwargs) -> cudaq.SampleResult:
        '''kwargs are passed to cudaq.sample(kernel, **kwargs)'''
        
        cudaq.set_target(self.cudaq_target, option=self.cudaq_target_option)
        if kwargs.get('shots_count', None) is None:
            self.samples_shots_count = 1000
        else:
            self.samples_shots_count = kwargs['shots_count']
        
        self.samples = cudaq.sample(self.kernel_hhl_complete, **kwargs)
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
        self.samples = cudaq.sample_async(self.kernel_hhl_complete, **kwargs)
        if self.verbose > 2:
            print('Finished sample_async')
        return None
    
    def get_state(self, **kwargs) -> cudaq.State:
        '''kwargs are passed to cudaq.get_state(kernel, **kwargs)'''
        cudaq.set_target(self.cudaq_target, option=self.cudaq_target_option)
        for meas in ['mz', 'mx', 'my']:
            if meas in self.string_kernel_hhl_complete:
                raise ValueError('Measurement in kernel_hhl_complete is not allowed, when requesting the quantum state')
        self.quantum_state = cudaq.get_state(self.kernel_hhl_complete, **kwargs)
        return self.quantum_state
    
    def get_state_async(self, **kwargs) -> None:
        '''kwargs are passed to cudaq.get_state(kernel, **kwargs)'''
        cudaq.set_target(self.cudaq_target, option=self.cudaq_target_option)
        for meas in ['mz', 'mx', 'my']:
            if meas in self.string_kernel_hhl_complete:
                raise ValueError('Measurement in kernel_hhl_complete is not allowed, when requesting the quantum state')
        self.quantum_state = cudaq.get_state_async(self.kernel_hhl_complete, **kwargs)
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