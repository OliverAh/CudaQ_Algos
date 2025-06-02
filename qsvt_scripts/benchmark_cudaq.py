print('Start imports')
import cudaq
import numpy as np
import time
print('Finished imports')

cudaq_target = 'nvidia'
cudaq_target_option = 'fp64'

shots_count = int(1e6)

num_qubits = 8
circuit_depth = 20000

a = 1/np.sqrt(2)*np.array([1.,1.,1.,-1.])
aa = 1/np.sqrt(2)*np.array([1.,1.,1.,-1.])
for i in range(num_qubits-2):
    aa = np.kron(aa, a)
print(aa.shape)
aa = aa.flatten()
print(aa.shape)
cudaq.register_operation('custom_h', 1/np.sqrt(2)*np.array([1.,1.,1.,-1.]))
cudaq.register_operation('custom_gate', aa)

@cudaq.kernel
def kernel():
    qvec = cudaq.qvector(num_qubits)
    
    for i in range(circuit_depth):
        for j in range(num_qubits):
            custom_h(qvec[j])
        #custom_gate.ctrl([qvec[7]], qvec[0], qvec[1], qvec[2], qvec[3], qvec[4], qvec[5], qvec[6])
        #for j in range(num_qubits-1):
        #    h.ctrl(qvec[0], qvec[j+1])
        h.ctrl([qvec[0]], qvec[1], qvec[2], qvec[3], qvec[4], qvec[5], qvec[6], qvec[7])

print('Compile kernel')
tic = time.time()
kernel.compile()
toc = time.time()
print('Finished compiling kernel in', f'{toc-tic}s')

cudaq.set_target(cudaq_target, option=cudaq_target_option)
tic = time.time()
samples = cudaq.sample(kernel, shots_count=shots_count)
toc = time.time()
print('Finished sampling in', f'{toc-tic}s')

cudaq.set_target(cudaq_target, option=cudaq_target_option)
#for meas in ['mz', 'mx', 'my']:
#    if meas in self.string_kernel_qsvt_complete:
#        raise ValueError('Measurement in kernel_qsvt_complete is not allowed, when requesting the quantum state')
tic = time.time()
quantum_state = cudaq.get_state(kernel)
toc = time.time()
print('Finished state computation in', f'{toc-tic}s')

print(cudaq.draw(kernel))