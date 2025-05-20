import cudaq

@cudaq.kernel
def circuit():
    qvec = cudaq.qvector(2)
    h(qvec[1])

print(cudaq.draw(circuit))
state = cudaq.get_state(circuit)
print(state)
state_amps = state.amplitudes(['00', '01', '10', '11'])
print(state_amps)
