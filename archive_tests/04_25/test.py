import cudaq

kernel = cudaq.make_kernel()
qubit = kernel.qalloc()
kernel.x(qubit)
kernel.mz(qubit)

result = cudaq.sample(kernel)

print(result)



@cudaq.kernel
def kernel(qubit_count: int):
    # Allocate our qubits.
    qvector = cudaq.qvector(qubit_count)
    # Place the first qubit in the superposition state.
    h(qvector[0])
    # Loop through the allocated qubits and apply controlled-X,
    # or CNOT, operations between them.
    for qubit in range(qubit_count):
        h(qvector[qubit])
    for qubit in range(qubit_count - 1):
        x.ctrl(qvector[qubit], qvector[qubit+1])
    for qubit in range(qubit_count):
        h(qvector[qubit])
    for qubit in range(qubit_count):
        ry(2.0, qvector[qubit])
        rz(1.0, qvector[qubit])
    for qubit in range(qubit_count):
        h(qvector[qubit])
    for qubit in range(qubit_count - 1):
        x.ctrl(qvector[qubit], qvector[qubit_count-1])
    # Measure the qubits.
    mz(qvector)
qubit_count = 4 # max on single gpu is 31
cudaq.set_target("nvidia", option = 'fp64')
#cudaq.set_target("qpp-cpu", option = 'fp64')
print(cudaq.draw(kernel, qubit_count))
results = cudaq.sample(kernel, qubit_count, shots_count=1000000)
print(results)

#targets = cudaq.get_targets()
#for t in targets:
#    print(t.name, t.description)
