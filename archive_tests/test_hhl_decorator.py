import cudaq
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pennylane as qml

from typing import List, Callable

import scipy.linalg

cudaq.set_target("nvidia", option="fp64")

def init_Ab(size:int=0):
    # Define alpha as the value before the b vector
    alpha = 12000*((5/7)**4)/(2e09 * 3.375e-4)

    A = np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            if i == j:
                A[i,j] = 6
            elif j == i+1 or j == i-1:
                A[i,j] = -4
            elif j == i-2 or j == i+2:
                A[i,j] = 1
            else:
                pass
        if i == 0 or i ==size-1:
            for j in range(size):
                A[i,j] = 0
    A[0,0] = 1
    A[size-1,size-1] = 1
    A[1,1] -= 1
    A[size-2,size-2] -= 1

    b = np.ones(size)
    b[0] = 0
    b[-1] = 0
    # Need to multiply right side vector by -alpha on exit!

    return A, b, alpha

def init_Ab_alt(size:int=6):
    if size != 6:
        print('Size must be 6')
        sys.exit(0)
    A = np.zeros((size,size))
    tmp = np.array([1, -4, 6, -4 , 1])
    #A[0,:5] = tmp
    A[1,1:] = tmp
    A[2,:5] = tmp
    A[3,1:] = tmp
    A[4,:5] = tmp
    #A[5,1:] = tmp

    A[0,:] = np.array([0, 0, 0, 0, 0, 0])
    A[:,0] = np.array([0, 0, 0, 0, 0, 0])
    A[-1,:] = np.array([0, 0, 0, 0, 0, 0])
    A[:,-1] = np.array([0, 0, 0, 0, 0, 0])
    A[0,0] = 1
    A[-1,-1] = 1
    
    b = np.ones(size)
    b[0] = 0
    b[-1] = 0

    return A, b

# Define a function to hermitianize the matrix
def hermitianize(mat):
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
            
# Check if matrix size is a power of 2, if not prepare matrix to size
# Check if the matrix is hermitian, if not Hermitianiz the matrix
def hermitian_matrix(A,b):
    if A.shape[0] != A.shape[1]:
        print('Not a Square Matrix')
        sys.exit(0)
    
    if int(np.log2(A.shape[0])) == float(np.log2(A.shape[0])):
        bs = int(np.log2(A.shape[0]))
        print('Matrix size is a power of 2. Base:', bs)
    else:
        print('Matrix size is not a power of 2')
        sys.exit(0)
    
    A_new = np.asmatrix(A)
    b_new = b
    if not scipy.linalg.ishermitian(A_new):
        print('Matrix A is not hermitian, system will be Hermitianized as [[A,0],[0,A^dagger]]')
        print(A_new)
        # Hermitianize the input matrix
        A_hermitian = hermitianize(A_new)
        # Adjust the right side vector with additional zeros
        b_hermitian = np.zeros(2*b_new.shape[0])
        b_hermitian[:b_new.shape[0]] = b_new
        # Change hermitian matrix data type from matrix to array
        A_hermitian = np.asarray(A_hermitian)
    else:
        print('Matrix A is hermitian, system is not changed')
        # Do nothing, the matrix is hermitian
        A_hermitian = np.asarray(A_new)
        b_adjust = b_new
    return A_hermitian, b_hermitian

#eye_bnum = np.eye(2**b_num)
#print('eye_bnum:', eye_bnum)
#cudaq.register_operation('init_b', 1/np.sqrt(b_num) * eye_bnum.flatten())
#cudaq.register_operation('init_b', 1/np.sqrt(4) * np.array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]))
#cudaq.register_operation("my_cnot", np.array([1, 0, 0, 0,
#                                              0, 1, 0, 0,
#                                              0, 0, 0, 1,
#                                              0, 0, 1, 0]))
#cudaq.register_operation('my_cnot', 1/np.sqrt(2**b_num) * eye_bnum.flatten())
eye_ccnot = np.eye(2**3)
cudaq.register_operation('my_ccnot', 1/np.sqrt(2**3) * eye_ccnot.flatten())





@cudaq.kernel
def HHL(A: List[float], b: List[float], r: int, t: float, eig_num: int, b_num: int, eye_bnum: List[float]):
    ##########
    # Determine required number of qubits
    ##########
    # Qubits
    #anc = QuantumRegister(1,'anc')
    #a_reg = QuantumRegister(eig_num, 'Eig_Reg')
    #b_reg = QuantumRegister(b_num, 'b_reg')
    num_anc = 1
    num_qubits = b_num + eig_num + num_anc
    qvec = cudaq.qvector(num_qubits) # b-, c-, anc-register
    # Classical Bits
    #cbit_anc = ClassicalRegister(1,'ancilla_cbit')
    #cbit_b = ClassicalRegister(b_num, 'regB')
    # Initialize circuit
    #circuit = QuantumCircuit(anc,a_reg,b_reg,cbit_anc,cbit_b)
    


    ##########
    # Prepare |b> in b register
    ##########
    # Initialize state b
    #init = Initialize(List(b_herm))
    #circuit.append(init,b_reg)
    #iiiinit_state(qvec, b, b_num, eye_bnum)

    my_cnot(qvec[num_anc + eig_num +0], qvec[num_anc + eig_num +1])

    my_ccnot(qvec[num_anc + eig_num -1], qvec[num_anc + eig_num +0], qvec[num_anc + eig_num +1])

    #iiiinit_state(qvec, b, b_num, eye_bnum)

    ##########
    # Prepare |Psi_0> in c register
    ##########
    
    for i in range(eig_num):
        h(qvec[b_num + i])

    ##########
    # Hamiltonian simulation exp(iAt) applied to |b>
    ##########

    #print(cudaq.draw(_init_state(qvec[num_anc+eig_num:], b, b_num, eye_bnum)))
    #circuit.draw()
    # Apply H-gate on quantum register a
    #circuit.h(a_reg)
    for i in range(num_anc):
        h(qvec[i])
    for j in range(eig_num+b_num):
        h(qvec[num_anc+j])
    # Apply controlled Hamiltonian operators on quantum register b
    #for i in range(eig_num):
    #    time = t/(2**(eig_num-1))
    #    U = HamiltonianGate(A_herm, time)
    #    G = U.control(1)
    #    qubit = [i+1]+[eig_num+j+1 for j in range(b_num)]
    #    circuit.append(G,qubit)
    
    # Apply inverse Quantum Fourier Transform
    #iqft = QFT(eig_num, approximation_degree=0, do_swaps=True, inverse=True, name='IQFT')
    #circuit.append(iqft, a_reg)
    # Swap Qubits in quantum register A
    #G = SwapGate()
    #circuit.append(G,[a_reg[1],a_reg[eig_num-1]])
    
    
    # Applying the decimal representation of the clock qubits --> Works with scaling factor at output
    #for i in range(eig_num):
    #    theta = 2*math.asin(1/(2**i))
    #    U = RYGate(theta).control(1)
    #    circuit.append(U,[i+1,0])
    
    
    #=============================== Uncompute the Circuit =========================#
    # Swap qubits in quantum register A
    #G = SwapGate()
    #circuit.append(G,[a_reg[1], a_reg[eig_num-1]])
    ## Apply Quantum Fourier Transform
    #qft = QFT(eig_num, approximation_degree=0, do_swaps=True, inverse=False, name='QFT')
    #circuit.append(qft, a_reg)
    ## Apply inverse controlled Hamiltonian Operators
    #for i in range(eig_num-1, -1, -1):
    #    time = t/(2**(eig_num-1))
    #    U = HamiltonianGate(-A_herm, time)
    #    G = U.control(1)
    #    qubit = [i+1]+[eig_num+j+1 for j in range(b_num)]
    #    circuit.append(G,qubit)
    ## Apply H Gate on Quantum Register A
    #circuit.h(a_reg)
    ## Measure the qubits
    #circuit.measure(anc, cbit_anc)
    #circuit.measure(b_reg, cbit_b)
    ## Return constructed circuit
    #return qvec




system_size = 4
log_system_size = int(np.log2(system_size))
A, b, alpha = init_Ab(system_size)
print('A:', A, 'b:', b, 'alpha:',alpha)


w = np.linalg.solve(A,b)
print('-alpha * b:', -alpha*b)
print('w=-alpha A^1 b:', -alpha*w)

#A_alt, b_alt = init_Ab_alt(6)
#print('A_alt:', A_alt, 'b_alt:', b_alt)
#w_alt = np.linalg.solve(A_alt,b_alt)
#print('-alpha * b:', -alpha*b_alt)
#print('w=-alpha A^1 b:', -alpha*w_alt)

x = np.linspace(0,5,system_size)
plt.plot(x,w)
plt.savefig('classical_sol.png')

A_herm, b_herm = hermitian_matrix(A,b)
print('A_herm:', A_herm, 'b_herm:', b_herm)
b_herm = -alpha*b_herm
b_herm
print('b_herm:', b_herm)

system_size_herm = A_herm.shape[0]
log_system_size_herm = int(np.log2(system_size_herm))
print('System size:', system_size_herm, 'Log System size:', log_system_size_herm)

# Condition number and Eigenvalues
cond_A_herm = np.linalg.cond(A_herm)
eig_A_herm = np.linalg.eigvals(A_herm)
print('Cond. number A_herm:', cond_A_herm)
print('Eigvals A_herm:', eig_A_herm)


# Number of Qubits needed to hold eigenvalues
eig_num = eig_A_herm.shape[0]
# Number of Qubits needed to hold right side vector
b_num = int(np.log2(len(b_herm)))
print('eig_num:', eig_num, 'b_num:', b_num)
# Hamiltonian Parameter --> Defines the total time for the Hamiltonial simulations
t = (1 - 2**(-1*eig_num))*0.75*np.pi/4
t = float(t)
print('t:', t)
# Rotation parameter --> From paper on Quantum Circuit Design choose value between 5 and 6 for best results. We select 5 (for higher success rate)
r = int(5)
# New implementation via arcsin function to get theta values
# Total number of circuit runs
shots = 80000


@cudaq.kernel
def init_qb(num_qubits: int, b: List[float], qb: cudaq.qview):
    # currently assumes b_i = a for all i
    h(qb[0])
    x(qb[1])
    x.ctrl(qb[0], qb[1])  # CNOT gate applied with qb[0] as control
    ry(np.pi, qb[0])
    ry(np.pi, qb[1])
    

@cudaq.kernel
def check_init_db(num_qubits: int, b: List[float]):
    qb = cudaq.qvector(num_qubits)
    init_qb(num_qubits, b, qb)
print("check init_qb")
print(cudaq.get_state(check_init_db, b_num, b_herm))
print(cudaq.draw(check_init_db, b_num, b_herm))


def pauli_decomp(A: np.ndarray):
    coeffs = []
    paulis = []
    pd = qml.pauli_decompose(A,pauli=False, hide_identity=False)
    coeffs = pd.coeffs
    for op in pd.ops:
        paulis.append(''.join(c for c in str(op) if c.isupper()))
    return coeffs, paulis
    
    
pd_coeffs, pd_paulis = pauli_decomp(A_herm)
print(pd_coeffs)
print(pd_paulis)


@cudaq.kernel
def hamiltonian_simulation_time_step_pauli(qvec: cudaq.qview, t: float, r: int, pd_coeffs: List[float], pd_paulis: List[cudaq.pauli_word]):
    '''This function applies applies a hamiltonian simulation time step of the form exp(iAt). 
    Matrix A must be provided as a pauli decomoposition with a list coefficients and a list of pauli strings.
    Pauli strings must be in the form of 'XIY', 'ZII', etc.
    r is the number of times the pauli decomposition is applied. It is used to approximate the original A matrix according to the
    Lie product formula: (exp(iAt/r)exp(iBt/r))^r = exp(i(A+B)t) + O((t^2)/r)'''
    
    for _ in range(r):
        for i in range(len(pd_coeffs)):
            coeff = pd_coeffs[i] * t / r
            pauli = pd_paulis[i]


@cudaq.kernel
def check_hamiltonian_simulation_time_step_pauli(num_qubits: int, system_size: int, b: List[float], A: List[float],
                                                 pd_coeffs: List[float], pd_paulis: List[cudaq.pauli_word], t: float, r: int):
    qb = cudaq.qvector(num_qubits)
    hamiltonian_simulation_time_step_pauli(qb, t, r, pd_coeffs, pd_paulis)

#print("check hamiltonian_simulation_time_step_pauli")
#print(cudaq.sample(check_hamiltonian_simulation_time_step_pauli, 1, system_size_herm, b_herm.flatten(), A_herm.flatten(), 
#                   pd_coeffs, pd_paulis, 1.0, 1))
#print(cudaq.get_state(check_hamiltonian_simulation_time_step_pauli, 1, system_size_herm, b_herm.flatten(), A_herm.flatten(), 
#                   pd_coeffs, pd_paulis, 1.0, 1))
#print(cudaq.draw(check_hamiltonian_simulation_time_step_pauli, 1, system_size_herm, b_herm.flatten(), A_herm.flatten(), 
#                   pd_coeffs, pd_paulis, 1.0, 1))


def pauliword_to_matrix(pauliword: str):
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
    print(a)
    return a
with np.printoptions(precision=3):
    operator_names = []
    for i in range(len(pd_coeffs)):
        print(pd_coeffs[i], pd_paulis[i])
        a = pauliword_to_matrix(pd_paulis[i])
        mexp = scipy.linalg.expm(1j * pd_coeffs[i] * a)
        mexp_alt = np.cos(pd_coeffs[i]) * np.eye(a.shape[0]) + 1j * np.sin(pd_coeffs[i]) * a
        assert np.allclose(mexp, mexp_alt), 'some problem with the matrix exponential'
        operator_names.append('exp('+pd_paulis[i]+')')

cudaq.register_operation('man_exp_pauli', mexp.flatten())

print(cudaq.globalRegisteredOperations['man_exp_pauli'])

@cudaq.kernel
def check_pauliword_to_matrix():
    q = cudaq.qvector(4)
    man_exp_pauli.ctrl(q[0],q[1],q[2],q[3])
    #cudaq.globalRegisteredOperations['man_exp_pauli'](q[1],q[2],q[3])
    #cudaq.

#print(cudaq.draw(check_pauliword_to_matrix))

def construct_string_register_operations_paulis(pd_coeffs: List[float], pd_paulis: List[str], t: List[float], r: int, size_qvec_c: int):
    s = ''
    s += 'import cudaq\n'
    s += 'import numpy as np\n'
    ops_names = []
    for j in range(len(t)):
        for rr in range(r):
            for i in range(len(pd_coeffs)):
                _pauli = pd_paulis[i]
                a = pauliword_to_matrix(_pauli)
                for qc in range(size_qvec_c):
                    _coeff = pd_coeffs[i] * t[j] / r * 2**qc
                    mexp = scipy.linalg.expm(1j * _coeff * a)
                    mexp_alt = np.cos(_coeff) * np.eye(a.shape[0]) + 1j * np.sin(_coeff) * a
                    assert np.allclose(mexp, mexp_alt), 'some problem with the matrix exponential'
                    _ops_name = 'exp__'+_pauli+'_t'+str(j)+'r'+str(rr)+'c'+str(qc)
                    ops_names.append(_ops_name)
                    s += 'cudaq.register_operation(\''+_ops_name+'\', np.array(' + np.array2string(mexp.astype(np.complex128).flatten(),precision=16,floatmode='maxprec',formatter={'complex_kind': lambda x: f'{x:.16e}'},separator=',').replace('\n', '') + ', dtype=np.complex128))\n'

    return s, ops_names

def construct_string_kernel_hamiltonian_simulation_timestep_pauli(pd_paulis: List[str], t: List[float], r: int, size_qvec_c: int):
    s = ''
    s += 'import cudaq\n'
    s += 'import register_operations_paulis\n'
    s += '@cudaq.kernel\n'
    s += 'def kernel(qvec_c: cudaq.qview, qvec_t: cudaq.qview):\n'
    #s += '    qubit = cudaq.qubit()\n'
    for j in range(len(t)):
        for rr in range(r):
            for _pauli in pd_paulis:
                for qc in range(size_qvec_c):
                    _ops_name = 'exp__'+_pauli+'_t'+str(j)+'r'+str(rr)+'c'+str(qc)
                    #s += '    '+_ops_name+'.ctrl(qvec_c['+str(qc)+'], qvec_t)\n'
                    #s += '    '+_ops_name+'.ctrl(qubit, qvec_t)\n'
                    #s += '    h(qvec_c)\n'
                    s += '    h(qvec_c['+str(qc)+'])\n'
    s += '    exp__XII_t0r0c0(qvec_t[2],qvec_t[1],qvec_t[0])\n'
    return s


t, r, size_qvec_c = [1.0], 1, 1
ops_strs, ops_names = construct_string_register_operations_paulis(pd_coeffs, pd_paulis, t, r, size_qvec_c)
#print(ops_strs)
with open('register_operations_paulis.py', 'w') as f:
    f.write(ops_strs)
#import register_operations_paulis
print(list(cudaq.globalRegisteredOperations.keys()))

kernel_str = construct_string_kernel_hamiltonian_simulation_timestep_pauli(pd_paulis, t, r, size_qvec_c)
print(kernel_str)
with open('file_hamiltonian_simulation_time_step_pauli.py', 'w') as f:
    f.write(kernel_str)

import file_hamiltonian_simulation_time_step_pauli
print(file_hamiltonian_simulation_time_step_pauli.kernel.to_json())
print(cudaq.globalRegisteredOperations.keys())
#abcde = cudaq.PyKernel(kernel_str)
#abcde = cudaq.kernel(function=file_hamiltonian_simulation_time_step_pauli.kernel, verbose=True, module=file_hamiltonian_simulation_time_step_pauli, kernelName='nicename')
abcde = cudaq.PyKernelDecorator.from_json(file_hamiltonian_simulation_time_step_pauli.kernel.to_json())
#print(abcde)
abcde.compile()
#print(abcde)

for key, val in cudaq.globalKernelRegistry.items():
    print(key)#, val)

@cudaq.kernel
def kernel2(size_qvec_c: int, size_qvec_b: int):
    qvec_c = cudaq.qvector(size_qvec_c) 
    qvec_b = cudaq.qvector(size_qvec_b)
    h(qvec_b)
    kernel(qvec_c, qvec_b)
#kernel3 = kernel2.merge_kernel(cudaq.globalKernelRegistry['kernel'])
print(cudaq.draw(kernel2, 2, 3))

####################
# Finally throws the following error:
#    error: 'quake.custom_op' op global not found for custom op
#    error: failed to legalize operation 'quake.custom_op'
#    RuntimeError: cudaq::builder failed to JIT compile the Quake representation.
# This is because of nested kernels with custom operations. See also https://github.com/NVIDIA/cuda-quantum/issues/2485
####################