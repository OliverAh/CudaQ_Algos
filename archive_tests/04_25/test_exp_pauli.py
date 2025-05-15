import cudaq

cudaq.set_target("nvidia")

words = ['XYZ', 'IXX']
coefficients = [0.432, 0.324]

@cudaq.kernel
def kernel(coefficients: list[float], words: list[cudaq.pauli_word], q: cudaq.qview):
    #q = cudaq.qvector(3)

    for i in range(len(coefficients)):
        exp_pauli(coefficients[i], q, words[i])


@cudaq.kernel
def kernel_2(coefficients: list[float], words: list[cudaq.pauli_word]):
    q = cudaq.qvector(3)
    kernel(coefficients, words, q)
cudaq.sample(kernel_2, coefficients, words)
cudaq.get_state(kernel_2, coefficients, words)
cudaq.draw(kernel_2, coefficients, words)
