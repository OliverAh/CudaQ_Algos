import cudaq
import numpy as np

numpy_customCX = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=np.complex128)
s = ''
s += 'import cudaq\n'
s += 'import numpy as np\n'
s += '\n'
s += 'cudaq.register_operation(\'customCX\', np.array(' + np.array2string(numpy_customCX.flatten(),formatter={'complex_kind': lambda x: f'{x:.16e}'},separator=',').replace('\n', '') + ', dtype=np.complex128))\n'
s += '\n'
s += '@cudaq.kernel\n'
s += 'def niceKernelName(qvec: cudaq.qview):\n'
s += '    customCX(' + ','.join([f'qvec[{str(i)}]' for i in range(2)]) + ')\n'

with open('mwe_kernel_merge_custom_op.py', 'w') as f:
    f.write(s)

import mwe_kernel_merge_custom_op
imported_kernel = cudaq.PyKernelDecorator.from_json(mwe_kernel_merge_custom_op.niceKernelName.to_json())

@cudaq.kernel
def actualKernel():
    qvec = cudaq.qvector(2)
    h(qvec[0])
    imported_kernel(qvec)

print(cudaq.draw(actualKernel))
print(cudaq.sample(actualKernel))

print(cudaq.__version__)