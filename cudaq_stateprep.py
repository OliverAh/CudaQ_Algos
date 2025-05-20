print('Start imports')
import cudaq
import numpy as np
import math
print('Finished imports')

'''
Cudaq uses big endian order for the qubit register --> |q0 q1> = a0 |00> + a1 |10> + a2 |01> + a3 |11> = [a0 a1 a2 a3].T
Cudaq assembles state vector as |q0 q1> = |0 1> = |q1> kron |q0> [0 1].T kron [1 0].T = [0 0 1 0].T
--> (X kron I) (|q1> kron |q0>) = (X kron I) |q0 q1> = (X kron I) |00> = X|q1> kron |q0> = |1> kron |0> = [0 0 1 0].T


-->  |q0 q1> = |0 1> = q1 kron q0 = 0 kron 1 = 0
-->                                 1      0   0
-->                                            1
-->                                            0

'''
state_amps = np.array([0., 0., 1., 0.]).reshape((4,1)) #a2=1, q0=1, q1=0

@cudaq.kernel
def stateprep():
    # Allocate qubits
    q = cudaq.qvector(2)
    # Prepare the state |q0 q1> = |01>
    ry(0., q[0])
    #ry(math.pi, q[1])
    ry(math.pi, q[1])


cudaq.set_target("nvidia", option="fp64")

samples = cudaq.sample(stateprep, shots_count=int(1e6))
print(samples)
state = cudaq.get_state(stateprep)
print(state)
print(state.amplitudes(['10', '01']))