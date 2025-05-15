import cudaq

cudaq.set_target("nvidia", option="mqpu,fp64")
print('number of gpus found:', cudaq.num_available_gpus())
@cudaq.kernel
def kernel():
    q = cudaq.qvector(2)
    h(q)
    #cx(q[0], q[1])

shots_count = int(1e9)

# The following line fails with the error: "TypeError: sample() got an unexpected keyword argument 'qpu_id'"
#print(cudaq.sample(kernel, shots_count=shots_count, qpu_id=3))

# The following line works, but always runs on GPU 0
print(cudaq.sample(kernel, shots_count=shots_count))

# The following line works, and respects the qpu_id argument, but is asynchronous.
#samples_async = cudaq.sample_async(kernel, shots_count=shots_count, qpu_id=3)
#print(samples_async.get())