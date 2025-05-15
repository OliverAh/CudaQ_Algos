import cudaq

@cudaq.kernel
def kernel():
    q = cudaq.qvector(2)
    x(q[0])


samples = cudaq.sample(kernel, shots_count=1000)
print('samples:', samples)
state = cudaq.get_state(kernel)
print('state:', state)
print('i', '| bs_le:', '| bs_be:', '| state[i]:', '| state.amplitude(bs_be):')
for i in range(2**2):
    bit_string_little_endian = format(i, '0' + str(2) + 'b')
    bit_string_big_endian = bit_string_little_endian[::-1]
    print(i,                        ' ',
          bit_string_little_endian, '     ',
          bit_string_big_endian,    '     ',
          state[i],                 '         ',
          state.amplitude(bit_string_big_endian))
