import cudaq
import numpy as np
import scipy
import matplotlib.pyplot as plt

from typing import List


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
        b_adjust = b_new
        # Change hermitian matrix data type from matrix to array
        A_hermitian = np.asarray(A_hermitian)
    else:
        print('Matrix A is hermitian, system is not changed')
        # Do nothing, the matrix is hermitian
        A_hermitian = np.asarray(A_new)
        b_adjust = b_new
    return A_hermitian, b_adjust


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
def iiiinit_state(q: cudaq.qview, b: List[float], b_num: int, eye_bnum: List[float]):
    qq = cudaq.qvector(b_num)
    #init_b(qq[0], qq[1])
    my_cnot(qq[0], qq[1])

@cudaq.kernel
def _hamiltonian_gate(mat: List[float], time: float):
    q = cudaq.qvector(mat.shape[0])
    #U = cudaq.HamiltonianGate(A, time)
    #U(q)


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
A_alt, b_alt = init_Ab_alt(6)
print('A_alt:', A_alt, 'b_alt:', b_alt)

w = np.linalg.solve(A,b)
print('-alpha * b:', -alpha*b)
print('w=-alpha A^1 b:', -alpha*w)

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

# Condition number and Eigenvalues
cond_A_herm = np.linalg.cond(A_herm)
eig_A_herm = np.linalg.eigvals(A_herm)
print('Cond. number A_herm:', cond_A_herm)
print('Eigvals A_herm:', eig_A_herm)


# Number of Qubits needed to hold eigenvalues
eig_num = eig_A_herm.shape[0]
# Number of Qubits needed to hold right side vector
b_num = int(np.log2(len(b_herm)))
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
def init_qb(b: List[float]):
    # currently assumes b_i = a for all i
    qvector = cudaq.qvector(b)
    h(qvector)

b_state_to_pass = cudaq.get_state(init_qb, b_herm)
print('b', b)
print('b_herm:', b_herm)
print('b_state_to_pass:', b_state_to_pass)

@cudaq.kernel
def hhl(b: List[float], qb: cudaq.State):
    qs = cudaq.qvector(qb)
    h(qs)



@cudaq.kernel
def init_qb_reg(b: List[float], qb: cudaq.qview):
    # currently assumes b_i = a for all i
    h(qb)


@cudaq.kernel
def hhl_initialized(b: List[float], qb: cudaq.qview):
    ry(np.pi, qb)


@cudaq.kernel
def kernel(num_qubits: int, b: List[float]):
    q_reg_b = cudaq.qvector(num_qubits)
    init_qb_reg(b, q_reg_b)
    hhl_initialized(b, q_reg_b)


@cudaq.kernel
def create_and_init_b_reg(num_qubits:int, b: List[float]) -> cudaq.qvector:
    q_reg_b = cudaq.qvector(num_qubits)
    h(q_reg_b)
    return q_reg_b

@cudaq.kernel
def take_b_reg(log_system_size: int, b: List[float]):
    qb = create_and_init_b_reg(log_system_size, b)
    ry(np.pi, qb)

state = cudaq.get_state(hhl, b_herm.flatten(), b_state_to_pass)
print(state)
print(cudaq.draw(hhl, b_herm.flatten(), b_state_to_pass))

state2 = cudaq.get_state(kernel, log_system_size, b_herm.flatten())
print(state2)
print(cudaq.draw(kernel, log_system_size, b_herm.flatten()))

state3 = cudaq.get_state(take_b_reg, log_system_size, b_herm.flatten())
print(state3)
print(cudaq.draw(take_b_reg, log_system_size, b_herm.flatten()))