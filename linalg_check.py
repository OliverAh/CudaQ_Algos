import numpy as np

size = 4
# Define alpha as the value in front of b vector, i.e. Ax=alpha*b
alpha = 12000*((5/7)**4)/(2e09 * 3.375e-4)
A = np.zeros((size,size))

tmp = [-1, 2, -1]
A[0,0:2] = [2, -1]
A[-1,-2:] = [-1, 2]
for i in range(1,size-1):
    A[i,i-1:i+2] = tmp
#b = np.ones((size,1))
b = np.array([[0.],[0.],[1.],[1.]])
b /= np.linalg.norm(b)

# [0.46807631+3.66922041e-16j 0.62299662-1.22307347e-15j 0.54427451-4.89229388e-16j 0.31071697+9.17305103e-17j]
# [0.46718756+6.10798750e-15j 0.62287261-2.45700295e-15j 0.54483417-7.27760212e-15j 0.31132174-5.19828723e-15j]
# [0.46718756+6.10798750e-15j 0.62287261-2.45700295e-15j 0.54483417-7.27760212e-15j 0.31132174-5.19828723e-15j]

x = np.array([[0.22677868],[0.45355737],[0.68033605],[0.52915026]])
#x = np.array([[0.39],[0.58],[0.58],[0.39]])

bb = A@x[[0,1,2,3]]
print(bb/np.linalg.norm(bb))
bb = A@x[[3,1,2,0]]
print(bb/np.linalg.norm(bb))
bb = A@x[[0,2,1,3]]
print(bb/np.linalg.norm(bb))
bb = A@x[[3,2,1,0]]
print(bb/np.linalg.norm(bb))
sol = np.linalg.solve(A, b)
print(sol/np.linalg.norm(sol))

#print(np.linalg.eig(A))




A2 = np.array(
[[ 0.5527864  ,-0.2763932 ,  0.        ,  0.        ,  0.72151566,  0.27467311 , -0.13863111,  0.05277535],
 [-0.2763932  , 0.5527864 , -0.2763932 ,  0.        ,  0.27467311,  0.58288455 ,  0.32744845, -0.13863111],
 [ 0.         ,-0.2763932 ,  0.5527864 , -0.2763932 , -0.13863111,  0.32744845 ,  0.58288455,  0.27467311],
 [ 0.         , 0.        , -0.2763932 ,  0.5527864 ,  0.05277535, -0.13863111 ,  0.27467311,  0.72151566],
 [ 0.72151566 , 0.27467311, -0.13863111,  0.05277535, -0.5527864 ,  0.2763932  ,-0.         ,-0.        ],
 [ 0.27467311 , 0.58288455,  0.32744845, -0.13863111,  0.2763932 , -0.5527864  , 0.2763932  ,-0.        ],
 [-0.13863111 , 0.32744845,  0.58288455,  0.27467311, -0.        ,  0.2763932  ,-0.5527864  , 0.2763932 ],
 [ 0.05277535 ,-0.13863111,  0.27467311,  0.72151566, -0.        , -0.   ,0.2763932 , -0.5527864 ]]
)

b2 = np.array(
[[0.        ],
 [0.        ],
 [0.70710678],
 [0.70710678],
 [0.        ],
 [0.        ],
 [0.        ],
 [0.        ]])

#x2 = np.linalg.solve(A2, b2)
#print(x2/np.linalg.norm(x2))

xx2 = np.linalg.solve(A2[0:4,0:4], b2[0:4])
print(xx2/np.linalg.norm(xx2))



def convert_biglittle_endian_unitary(unitary: np.ndarray) -> np.ndarray:
    """
    Converts the unitary matrix of a multiqubit gate from little endian to big endian notation and vice versa.
    
    Args:
        unitary: A numpy array of a unitary matrix in either little or big endian notation
        
    Returns:
        A numpy array representing the unitary matrix in big or little endian notation
    
    Notes:
        For additional information see https://quantumcomputing.stackexchange.com/questions/26899/how-to-convert-between-little-big-endian-unitary-forms-in-braket
    """
    qubit_count = int(np.log2(unitary.shape[0]))
    U_tensor = unitary.reshape([2] * 2 * qubit_count)
    input = list(reversed(range(qubit_count)))
    output = [i + qubit_count for i in input]
    biglittle_endian_tensor = np.einsum(U_tensor, input + output)
    return biglittle_endian_tensor.reshape([2 ** qubit_count, 2 ** qubit_count])
    
    #

U = A2

with np.printoptions(precision=3, linewidth=200):
    print(U)
    U_big_endian = convert_biglittle_endian_unitary(U)
    print(U_big_endian)

    U_little_endian = convert_biglittle_endian_unitary(U_big_endian)
    print(U_little_endian)

    print(U - U_little_endian)
    print(np.allclose(U, U_little_endian))