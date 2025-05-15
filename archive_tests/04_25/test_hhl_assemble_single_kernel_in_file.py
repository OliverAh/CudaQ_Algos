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

def init_Ab_alt(size:int=4):
    alpha = 12000*((5/7)**4)/(2e09 * 3.375e-4)
    A = np.zeros((size,size))
    
    tmp = [-1, 2, -1]
    A[0,0:2] = [2, -1]
    A[-1,-2:] = [-1, 2]
    for i in range(1,size-1):
        A[i,i-1:i+2] = tmp
    b = np.ones(size)
    
    return A, b, alpha

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
        b_hermitian = b_new
    return A_hermitian, b_hermitian


system_size = 4
log_system_size = int(np.log2(system_size))
A, b, alpha = init_Ab_alt(system_size)
print('A:\n', A, 'b:', b, 'alpha:',alpha)

w = np.linalg.solve(A,b)
print('alpha * b:', alpha*b)
print('w=alpha A^1 b:', alpha*w)

x = np.linspace(0,5,system_size)
plt.plot(x,w, label='classical')
plt.savefig('classical_sol.png')

A_herm, b_herm = hermitian_matrix(A,b)
print('A_herm:', A_herm, 'b_herm:', b_herm)
b_herm = alpha*b_herm
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


@cudaq.kernel
def init_qb(num_qubits: int, b: List[float], qb: cudaq.qview):
    # currently assumes b_i = a for all i
    h(qb[0])
    x(qb[1])
    x.ctrl(qb[0], qb[1])  # CNOT gate applied with qb[0] as control
    ry(np.pi, qb[0])
    ry(np.pi, qb[1])

@cudaq.kernel
def init_qb_alt(num_qubits: int, b: List[float], qb: cudaq.qview):
    # currently assumes b_i = a for all i
    h(qb[0])
    x(qb[1])
    x.ctrl(qb[0], qb[1])  # CNOT gate applied with qb[0] as control
    ry(np.pi, qb[0])
    ry(np.pi, qb[1])

def pauli_decomp(A: np.ndarray):
    coeffs = []
    paulis = []
    pd = qml.pauli_decompose(A,pauli=False, hide_identity=False)
    coeffs = pd.coeffs
    for op in pd.ops:
        paulis.append(''.join(c for c in str(op) if c.isupper()))
    return coeffs, paulis
    
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
    #print(a)
    return a

#pd_coeffs, pd_paulis = pauli_decomp(A_herm)
#print(pd_coeffs)
#print(pd_paulis)
#with np.printoptions(precision=3):
#    operator_names = []
#    for i in range(len(pd_coeffs)):
#        #print(pd_coeffs[i], pd_paulis[i])
#        a = pauliword_to_matrix(pd_paulis[i])
#        mexp = scipy.linalg.expm(1j * pd_coeffs[i] * a)
#        mexp_alt = np.cos(pd_coeffs[i]) * np.eye(a.shape[0]) + 1j * np.sin(pd_coeffs[i]) * a
#        assert np.allclose(mexp, mexp_alt), 'some problem with the matrix exponential'
#        operator_names.append('exp('+pd_paulis[i]+')')
#cudaq.register_operation('man_exp_pauli', mexp.flatten())
#print(cudaq.globalRegisteredOperations['man_exp_pauli'])

@cudaq.kernel
def check_pauliword_to_matrix():
    q = cudaq.qvector(4)
    man_exp_pauli.ctrl(q[0],q[1],q[2],q[3])
    #cudaq.globalRegisteredOperations['man_exp_pauli'](q[1],q[2],q[3])
    #cudaq.

def construct_string_register_operations_paulis(pd_coeffs: List[float], pd_paulis: List[str], t: List[float], r: int, size_qvec_c: int):
    s = ''
    #s += 'import cudaq\n'
    #s += 'import numpy as np\n'
    ops_names = []
    for j in range(len(t)):
        for rr in range(r):
            for i in range(len(pd_coeffs)):
                _pauli = pd_paulis[i]
                a = pauliword_to_matrix(_pauli)
                for qc in range(size_qvec_c):
                    _coeff = pd_coeffs[i] * t[j] / r * 2**qc
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

    return s, ops_names

def construct_string_kernel_initialization_b_register():
    # currently assumes b_i = a for all i
    s = ''
    s += '    h(qvec_b[0])\n'
    s += '    x(qvec_b[1])\n'
    s += '    x.ctrl(qvec_b[0], qvec_b[1])  # CNOT gate applied with qb[0] as control\n'
    s += '    ry(np.pi, qvec_b[0])\n'
    s += '    ry(np.pi, qvec_b[1])\n'
    return s

def construct_string_kernel_hamiltonian_simulation_timestep_pauli(pd_paulis: List[str], t: List[float], r: int, size_qvec_c: int, size_qvec_b: int):
    s = ''
    #s += 'import cudaq\n'
    #s += 'import register_operations_paulis\n'
    #s += '@cudaq.kernel\n'
    #s += 'def kernel(qvec_c: cudaq.qview, qvec_t: cudaq.qview):\n'
    #s += '    qubit = cudaq.qubit()\n'
    for j in range(len(t)):
        for rr in range(r):
            for _pauli in pd_paulis:
                for qc in range(size_qvec_c-1,-1,-1):
                    _ops_name = 'exp__'+_pauli+'_t'+str(j)+'r'+str(rr)+'c'+str(qc)
                    s += '    '+_ops_name+'.ctrl(qvec_c['+str(qc)+'],' + ','.join([f'qvec_b[{str(i)}]' for i in range(size_qvec_b)]) + ')\n'
                    #s += '    '+_ops_name+'.ctrl(qubit, qvec_t)\n'
                    #s += '    h(qvec_c)\n'
                    #s += '    h(qvec_c['+str(qc)+'])\n'
    #s += '    exp__XII_t0r0c0(qvec_t[2],qvec_t[1],qvec_t[0])\n'
    return s

def construct_string_kernel_hamiltonian_simulation_timestep_pauli_dagger(pd_paulis: List[str], t: List[float], r: int, size_qvec_c: int, size_qvec_b: int):
    s = ''
    for j in range(len(t)-1,-1,-1):
        for rr in range(r-1,-1,-1):
            for _pauli in pd_paulis:
                for qc in range(size_qvec_c):
                    _ops_name = 'exp__'+_pauli+'_t'+str(j)+'r'+str(rr)+'c'+str(qc)
                    _ops_name = 'adj_'+_ops_name
                    s += '    '+_ops_name+'.ctrl(qvec_c['+str(qc)+'],' + ','.join([f'qvec_b[{str(i)}]' for i in range(size_qvec_b)]) + ')\n'
                    #s += '    h(qvec_c)\n'
    return s

def construct_string_kernel_qft(size_qvec_c: int):
    s = ''
    for i in range(size_qvec_c):
        s += '    h(qvec_c['+str(i)+'])\n'
        for j in range(i + 1, size_qvec_c):
            #angle = (2 * np.pi) / (2**(j - i + 1)) # why +1?
            angle = (2 * np.pi) / (2**(j - i))
            s += f'    cr1({angle}, [qvec_c[{j}]], qvec_c[{i}])\n'
    for i in range(size_qvec_c):
        a = (size_qvec_c - 1) - i
        b = i
        if not (a == b) and not (a < b):
            s += f'    swap(qvec_c[{a}], qvec_c[{b}])\n'
    return s

def construct_string_kernel_qft_dagger(size_qvec_c: int):
    s = ''
    swaps = []
    for i in range(size_qvec_c):
        a = (size_qvec_c - 1) - i
        b = i
        if not (a == b) and not (a < b):
            swaps.append((a,b))
    if not len(swaps) == 0:
        swaps.reverse()
        print(swaps)
        for a,b in swaps:
            print('a,b', a,b)
            s += f'    swap(qvec_c[{a}], qvec_c[{b}])\n'
    for i in range(size_qvec_c-1,-1,-1):
        for j in range(size_qvec_c-1, i, -1):
            angle = -(2 * np.pi) / (2**(j - i + 1))
            s += f'    cr1({angle}, [qvec_c[{j}]], qvec_c[{i}])\n'
        s += '    h(qvec_c['+str(i)+'])\n'
    return s

def construct_string_kernel_ancilla_rotation(size_qvec_c: int):
    s = ''
    for i in range(1,size_qvec_c):
        angle = 2*np.asin(1/(2**i))
        s += f'    ry.ctrl({angle},qvec_c['+str(i)+'],qbit_a)\n'
    return s

def construct_string_hhl_complete(size_qvec_c: int, size_qvec_b: int, t: List[float], r: int, pd_coeffs: List[float], pd_paulis: List[str], b_herm: np.ndarray, A_herm: np.ndarray):
    s_hhl_complete = ''
    s_hhl_complete += 'import cudaq\n'
    s_hhl_complete += 'import numpy as np\n\n'

    s_operations_paulis, operations_paulis_names = construct_string_register_operations_paulis(pd_coeffs, pd_paulis, t, r, size_qvec_c)
    s_hhl_complete += s_operations_paulis + '\n'

    s_hhl_complete += '@cudaq.kernel\n'
    s_hhl_complete += 'def hhl():\n'
    s_hhl_complete += '    qbit_a = cudaq.qubit()\n'
    s_hhl_complete += '    qvec_c = cudaq.qvector('+str(size_qvec_c)+')\n'
    s_hhl_complete += '    qvec_b = cudaq.qvector('+str(size_qvec_b)+')\n'

    s_hhl_complete += '\n'
    s_hhl_complete += '    ####################\n'
    s_hhl_complete += '    # init b register\n'
    s_hhl_complete += '    ####################\n'
    s_hhl_complete += '\n'
    
    s_initialization_b_register = construct_string_kernel_initialization_b_register()
    s_hhl_complete += s_initialization_b_register + '\n'

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

    s_hamiltonian_simulation_time_step_pauli = construct_string_kernel_hamiltonian_simulation_timestep_pauli(pd_paulis, t, r, size_qvec_c, size_qvec_b)
    s_hhl_complete += s_hamiltonian_simulation_time_step_pauli + '\n'
    #s_hhl_complete += '    h(qvec_b)\n'
    #s_hhl_complete += '    h(qvec_c)\n'

    s_hhl_complete += '\n'
    s_hhl_complete += '    ####################\n'
    s_hhl_complete += '    # apply qft as part of qpe\n'
    s_hhl_complete += '    ####################\n'
    s_hhl_complete += '\n'

    s_qft = construct_string_kernel_qft(size_qvec_c)
    s_hhl_complete += s_qft + '\n'


    s_hhl_complete += '\n'
    s_hhl_complete += '    ####################\n'
    s_hhl_complete += '    # apply controlled ancilla rotation\n'
    s_hhl_complete += '    ####################\n'
    s_hhl_complete += '\n'
    
    s_ancilla_rotation = construct_string_kernel_ancilla_rotation(size_qvec_c)
    s_hhl_complete += s_ancilla_rotation + '\n'




    s_hhl_complete += '\n'
    s_hhl_complete += '    ####################\n'
    s_hhl_complete += '    # apply qft_dagger as part of qpe_dagger\n'
    s_hhl_complete += '    ####################\n'
    s_hhl_complete += '\n'

    s_qft_dagger = construct_string_kernel_qft_dagger(size_qvec_c)
    s_hhl_complete += s_qft_dagger + '\n'


    s_hhl_complete += '\n'
    s_hhl_complete += '    ####################\n'
    s_hhl_complete += '    # apply hamiltonian simulation dagger as part of qpe_dagger\n'
    s_hhl_complete += '    ####################\n'
    s_hhl_complete += '\n'

    s_hamiltonian_simulation_time_step_pauli_dagger = construct_string_kernel_hamiltonian_simulation_timestep_pauli_dagger(pd_paulis, t, r, size_qvec_c, size_qvec_b)
    s_hhl_complete += s_hamiltonian_simulation_time_step_pauli_dagger + '\n'
    

    s_hhl_complete += '\n'
    s_hhl_complete += '    ####################\n'
    s_hhl_complete += '    # init_dagger c register\n'
    s_hhl_complete += '    ####################\n'
    s_hhl_complete += '\n'

    s_hhl_complete += '    h(qvec_c)\n'

    s_hhl_complete += '\n'
    s_hhl_complete += '    ####################\n'
    s_hhl_complete += '    # measure b register\n'
    s_hhl_complete += '    ####################\n'
    s_hhl_complete += '\n'
    
    s_hhl_complete += '    mz(qvec_b)\n'

    return s_hhl_complete


pd_coeffs, pd_paulis = pauli_decomp(A_herm)
#print(pd_coeffs)
#print(pd_paulis)

# Number of eigenvalues
eig_num = eig_A_herm.shape[0]
# Number of Qubits needed to hold right side vector
b_num = int(np.log2(len(b_herm)))
print('eig_num:', eig_num, 'b_num:', b_num)
# Hamiltonian Parameter --> Defines the total time for the Hamiltonial simulations
#t = (1 - 2**(-1*eig_num))*0.75*np.pi/4
#t = float(t)
#print('t:', t)
# Rotation parameter --> From paper on Quantum Circuit Design choose value between 5 and 6 for best results. We select 5 (for higher success rate)
#r = int(5)
# New implementation via arcsin function to get theta values
# Total number of circuit runs
#shots = 80000

size_qvec_c = system_size_herm
size_qvec_b = int(np.log2(b_herm.shape[0]))
bit_accuracy = 4
log_bit_accuracy = int(np.log2(bit_accuracy))
error = 1e-1
t = [1.0, 2.0]
t = [np.max(eig_A_herm)]
#t : 2pi A t = phi
#t = [1/(2*np.pi*eig) for eig in eig_A_herm]
#t = [(1 + log_bit_accuracy + 1/(4*error))]
#t = [(1 - 2**(-1*eig_num))*0.75*np.pi/4]
print('t:', t)
r = 5
s_hhl_complete = construct_string_hhl_complete(size_qvec_c, size_qvec_b, t, r, pd_coeffs, pd_paulis, b_herm, A_herm)
with open('kernels/kernel_hhl_complete.py', 'w') as f:
    f.write(s_hhl_complete)


from kernels import kernel_hhl_complete
imported_kernel = cudaq.PyKernelDecorator.from_json(kernel_hhl_complete.hhl.to_json())

#print(cudaq.draw(imported_kernel))
#print(cudaq.get_state(imported_kernel))
#print(cudaq.sample(imported_kernel))

@cudaq.kernel
def actualKernel():
    imported_kernel()

shots_count = int(1e7)
print(cudaq.draw(actualKernel))
state = cudaq.get_state(actualKernel)
print(state)
samples = cudaq.sample(actualKernel, shots_count=shots_count)
print('samples:', samples)
samples_count = np.array(list(samples.values()))
print('samples_count:', samples_count)
print('samples_count[1]/samples_count[0]:', samples_count[1]/samples_count[0])
print('w[1]/w[0]:', w[1]/w[0])
print('max(samples_count)/min(samples_count):', np.max(samples_count)/np.min(samples_count))
print('max(w)/min(w):', np.max(w)/np.min(w))
quantum_solution = samples_count / shots_count
print('samples normalized:', quantum_solution)
print('ratio Ax / b:', [(A_herm@quantum_solution)[i] / b_herm[i] for i in range(len(b_herm))])
print('Ax, b:', A_herm@quantum_solution, b_herm)
scaling_factor = np.average([(A_herm@quantum_solution)[i] / b_herm[i] for i in range(len(b_herm))])
print('scaling_factor:', scaling_factor)
quantum_solution *= scaling_factor
print(quantum_solution)


plt.plot(x, quantum_solution, label='quantum')
#plt.plot(x,1/np.linalg.norm(b_herm)/shots_count*np.array(list(samples.values())), label='quantum')
plt.legend()
plt.savefig('classical_sol.png')
print(1/np.linalg.norm(b_herm))


#print(dir(someKernel))
#print(someKernel)
#print(imported_kernel)


def check_hamiltonian_sim_unitary(size_qvec_c: int, size_qvec_b: int, t: List[float], r: int, pd_coeffs: List[float], pd_paulis: List[str], b_herm: np.ndarray, A_herm: np.ndarray):
    s = ''
    s = ''
    s += 'import cudaq\n'
    s += 'import numpy as np\n\n'

    s_operations_paulis, operations_paulis_names = construct_string_register_operations_paulis(pd_coeffs, pd_paulis, t, r, size_qvec_c)
    s += s_operations_paulis + '\n'

    s += '@cudaq.kernel\n'
    s += 'def check_hamiltonian_sim_unitary():\n'
    s += '    qbit_a = cudaq.qubit()\n'
    s += '    qvec_c = cudaq.qvector('+str(size_qvec_c)+')\n'
    s += '    qvec_b = cudaq.qvector('+str(size_qvec_b)+')\n'



    s += '\n'
    s += '    ####################\n'
    s += '    # apply hamiltonian simulation as part of qpe\n'
    s += '    ####################\n'
    s += '\n'

    s_hamiltonian_simulation_time_step_pauli = construct_string_kernel_hamiltonian_simulation_timestep_pauli(pd_paulis, t, r, size_qvec_c, size_qvec_b)
    s += s_hamiltonian_simulation_time_step_pauli + '\n'


    s += '\n'
    s += '    ####################\n'
    s += '    # apply hamiltonian simulation dagger as part of qpe_dagger\n'
    s += '    ####################\n'
    s += '\n'

    s_hamiltonian_simulation_time_step_pauli_dagger = construct_string_kernel_hamiltonian_simulation_timestep_pauli_dagger(pd_paulis, t, r, size_qvec_c, size_qvec_b)
    s += s_hamiltonian_simulation_time_step_pauli_dagger + '\n'

    with open('kernels/check_hamiltonian_sim_unitary.py', 'w') as f:
        f.write(s)

    from kernels import check_hamiltonian_sim_unitary
    imported_kernel = cudaq.PyKernelDecorator.from_json(check_hamiltonian_sim_unitary.check_hamiltonian_sim_unitary.to_json())

    @cudaq.kernel
    def actualKernel():
        imported_kernel()

    print(cudaq.draw(actualKernel))
    print(cudaq.get_state(actualKernel))
    print(cudaq.sample(actualKernel))

#check_hamiltonian_sim_unitary(size_qvec_c, size_qvec_b, t, r, pd_coeffs, pd_paulis, b_herm, A_herm)

def verify_qft_and_qft_dagger_and_unitary(size_qvec_c: int):
    '''for size_qvec_c = 3, and initial state 101, the qft should return
      [ 0.35+0.j   -0.25-0.25j  0.  +0.35j  0.25-0.25j -0.35+0.j    0.25+0.25j  -0.-0.35j -0.25+0.25j]'''
    s = ''
    s = ''
    s += 'import cudaq\n'
    s += 'import numpy as np\n\n'

    s += '@cudaq.kernel\n'
    s += 'def verify_qft_and_qft_dagger_and_unitary():\n'
    #s += '    qbit_a = cudaq.qubit()\n'
    s += '    qvec_c = cudaq.qvector('+str(size_qvec_c)+')\n'
    #s += '    qvec_b = cudaq.qvector('+str(size_qvec_b)+')\n'

    s += '\n'
    s += '    ####################\n'
    s += '    # init qvec_c as 101\n'
    s += '    ####################\n'
    s += '\n'

    s += '    x(qvec_c[0])\n'
    s += '    x(qvec_c[2])\n'

    s += '\n'
    s += '    ####################\n'
    s += '    # apply qft as part of qpe\n'
    s += '    ####################\n'
    s += '\n'

    s_qft = construct_string_kernel_qft(size_qvec_c)
    s += s_qft + '\n'


    #s += '\n'
    #s += '    ####################\n'
    #s += '    # apply qft_dagger as part of qpe_dagger\n'
    #s += '    ####################\n'
    #s += '\n'
    #
    #s_qft_dagger = construct_string_kernel_qft_dagger(size_qvec_c)
    #s += s_qft_dagger + '\n'


    with open('kernels/verify_qft_and_qft_dagger_and_unitary.py', 'w') as f:
        f.write(s)

    from kernels import verify_qft_and_qft_dagger_and_unitary
    imported_kernel = cudaq.PyKernelDecorator.from_json(verify_qft_and_qft_dagger_and_unitary.verify_qft_and_qft_dagger_and_unitary.to_json())

    @cudaq.kernel
    def actualKernel():
        imported_kernel()

    print(cudaq.draw(actualKernel))
    print(cudaq.get_state(actualKernel))
    print(cudaq.sample(actualKernel))

#verify_qft_and_qft_dagger_and_unitary(3)

#def verify_hamitonian_sim_and_dagger_and_unitary():