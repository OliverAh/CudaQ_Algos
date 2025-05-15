#import cudaq
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent))
import matplotlib.pyplot as plt

import numpy as np
import pyqsp
import pyqsp.angle_sequence
import pennylane as qml


from src import qsvt



A = np.array(
   [
       [0.65713691, -0.05349524, 0.08024556, -0.07242864],
       [-0.05349524, 0.65713691, -0.07242864, 0.08024556],
       [0.08024556, -0.07242864, 0.65713691, -0.05349524],
       [-0.07242864, 0.08024556, -0.05349524, 0.65713691],
   ]
)
b = np.array([1., 2., 3., 4.]).reshape((4,1))

#qsvt_instance = qsvt.QSVT(A = A,
#                          b = b,
qsvt_instance = qsvt.QSVT(
                         cudaq_target = 'nvidia',
                         cudaq_target_option = 'fp64',
                         verbose=99)
qsvt_instance.hermitianize_system()
qsvt_instance.BlockEncode_A_unitary()
cond = qsvt_instance._compute_condition_number()
print('Condition number:', cond)
#with np.printoptions(precision=2, linewidth=200):
#    print(qsvt_instance.A_block_encoded_unitary)


kappa = cond
poly_oneoverx, scale_oneoverx = pyqsp.poly.PolyOneOverX().generate(kappa=kappa, return_coef=True, ensure_bounded=True, return_scale=True)
#print(poly_oneoverx)
#print(scale_oneoverx)

angles_poly_oneoverx = pyqsp.angle_sequence.QuantumSignalProcessingPhases(poly_oneoverx, signal_operator="Wx", tolerance=0.00001)
phi_qsvt = qml.transform_angles(angles_poly_oneoverx, "QSP", "QSVT")
#print(phi_qsvt)
print('Degree of polynomial:', len(poly_oneoverx)-1, len(angles_poly_oneoverx)-1, len(phi_qsvt)-1)
x_vals = np.linspace(0, 1, 50)
target_y_vals = [scale_oneoverx * (1 / x) for x in np.linspace(scale_oneoverx, 1, 50)]

qsvt_y_vals = []
for x in x_vals:

    block_encoding = qml.BlockEncode(x, wires=[0])
    projectors = [qml.PCPhase(angle, dim=1, wires=[0]) for angle in phi_qsvt]

    poly_x = qml.matrix(qml.QSVT, wire_order=[0])(block_encoding, projectors)
    qsvt_y_vals.append(np.real(poly_x[0, 0]))

plt.plot(x_vals, np.array(qsvt_y_vals), label="Re(qsvt)")
plt.plot(np.linspace(scale_oneoverx, 1, 50), target_y_vals, label="target")

plt.vlines(1 / kappa, -1.0, 1.0, linestyle="--", color="grey", label="1/kappa")
plt.vlines(0.0, -1.0, 1.0, color="black")
plt.hlines(0.0, -0.1, 1.0, color="black")

plt.legend()
plt.show()
plt.savefig("qsvt_oneoverx.png", dpi=300)
#polytaylor, scale = pyqsp.poly.PolyTaylorSeries().taylor_series(lambda x: 1/x, degree=43, chebyshev_basis=False, return_scale=True)
#print(polytaylor)
#print(scale)




qsvt_y_vals2 = []
#block_encoding2 = qsvt_instance.A_block_encoded_unitary
projectors2 = [qml.PCPhase(angle, dim=4, wires=[1,2,3]) for angle in phi_qsvt]
projectors3 = projectors2#[qml.PCPhase(0.0, dim=4, wires=[1,2,3]), qml.PCPhase(0.0, dim=4, wires=[1,2,3]), qml.PCPhase(0.0, dim=4, wires=[1,2,3])]
pennylane_blockencoding = qml.BlockEncode(qsvt_instance.A_hermitian, wires=[1,2,3])
with np.printoptions(precision=2, linewidth=200):
    print(projectors2[0].matrix())
    print(qsvt_instance.A_block_encoded_unitary)
    print(pennylane_blockencoding.matrix())
    

q_script = qml.tape.QuantumScript(ops=[qml.QSVT(pennylane_blockencoding, projectors3)])
#print(q_script.expand().draw(decimals=2))
matrix_qsvt_pennylane = qml.matrix(qml.QSVT, wire_order=[1,2,3])(pennylane_blockencoding, projectors3)
print('qsvt_pennylane')
with np.printoptions(precision=3, linewidth=200):
    print(matrix_qsvt_pennylane)


with qml.queuing.AnnotatedQueue() as q:
    qml.PCPhase(phi_qsvt[0], dim=4, wires=[1,2,3])
    for i in range(1,len(projectors2)):
        qml.BlockEncode(qsvt_instance.A_hermitian, wires=[1,2,3])
        qml.PCPhase(phi_qsvt[i], dim=4, wires=[1,2,3])
    
q_script2 = qml.tape.QuantumScript.from_queue(q)
#print(q_script2.draw(decimals=2))
print('qsvt_m')
matrix_qsvt_m = qml.matrix(q_script2, wire_order=[1,2,3])
matrix_qsvt_m_adj = qml.matrix(q_script2, wire_order=[1,2,3]).conjugate().transpose()
with np.printoptions(precision=3, linewidth=200):
    b = qsvt_instance.b_block_encoded_normalized
    #b = np.zeros((matrix_qsvt_m.shape[0],1))
    #b[:4] = np.array([1, 2, 3, 4]).reshape((4,1))
    #b /= np.linalg.norm(b)
    q_x = 0.5*(matrix_qsvt_m + matrix_qsvt_m_adj) @ b
    q_x = q_x[:4]
    q_x /= np.linalg.norm(q_x)
    print(0.5*(matrix_qsvt_m + matrix_qsvt_m_adj))
    print(q_x)


is_close_qsvr_pennylane_m = np.allclose(matrix_qsvt_m, matrix_qsvt_pennylane)
if is_close_qsvr_pennylane_m:
    print('qsvt_m and qsvt_pennylane are close')
else:
    print('qsvt_m and qsvt_pennylane are NOT close')
    with np.printoptions(precision=2, linewidth=200):
        print('Difference:', matrix_qsvt_m - matrix_qsvt_pennylane)

with np.printoptions(precision=3, linewidth=200):
    print(qsvt_instance.A_hermitian_unitary)
    print(np.linalg.eigvals(qsvt_instance.A_hermitian_unitary))
    b = qsvt_instance.b
    #b = np.array([1., 2., 3., 4.]).reshape((4,1))
    b /= np.linalg.norm(b)
    target_x = np.linalg.inv(qsvt_instance.A) @ b
    target_x /= np.linalg.norm(target_x)
    target_x2 = qsvt_instance.A_hermitian_unitary_inverse @ b
    target_x2 /= np.linalg.norm(target_x2)
    print(target_x)
    print(target_x2)
    
with qml.queuing.AnnotatedQueue() as q2:
    qml.Identity(wires=[0,1,2,3])#0is ancilla qubit
    qml.Hadamard(wires=[0])
    qml.ctrl(qml.PCPhase(phi_qsvt[0], dim=4, wires=[1,2,3]), control=(0,), control_values=(0,))
    for i in range(1,len(projectors2)):
        qml.ctrl(qml.BlockEncode(qsvt_instance.A_hermitian, wires=[1,2,3]), control=(0,), control_values=(0,))
        qml.ctrl(qml.PCPhase(phi_qsvt[i], dim=4, wires=[1,2,3]), control=(0,), control_values=(0,))
    
    for i in range(len(projectors2)-1, 0,-1):
    #for i in range(1,len(projectors2)):
        qml.ctrl(qml.adjoint(qml.PCPhase(phi_qsvt[i], dim=4, wires=[1,2,3])), control=(0,), control_values=(1,))
        qml.ctrl(qml.adjoint(qml.BlockEncode(qsvt_instance.A_hermitian, wires=[1,2,3])), control=(0,), control_values=(1,))
    qml.ctrl(qml.adjoint(qml.PCPhase(phi_qsvt[0], dim=4, wires=[1,2,3])), control=(0,), control_values=(1,))
    
    #for i in range(len(projectors2)-1, -1,-1):
    # qml.adjoint(qml.ctrl(qml.PCPhase(phi_qsvt[0], dim=4, wires=[1,2,3]), control=(0,), control_values=(1,)))
    # for i in range(1,len(projectors2)):
        # qml.adjoint(qml.ctrl(qml.BlockEncode(qsvt_instance.A_hermitian, wires=[1,2,3]), control=(0,), control_values=(1,)))
        # qml.adjoint(qml.ctrl(qml.PCPhase(phi_qsvt[i], dim=4, wires=[1,2,3]), control=(0,), control_values=(1,)))

    qml.Hadamard(wires=[0])
    
q_script_controlled_qsvt = qml.tape.QuantumScript.from_queue(q2)
print('qsvt_m_controlled')
#print(q_script_controlled_qsvt.draw(decimals=2))
matrix_qsvt_m_controlled = qml.matrix(q_script_controlled_qsvt, wire_order=[0,1,2,3])
with np.printoptions(precision=3, linewidth=200):
    print(matrix_qsvt_m_controlled[:8,:8])
    print(matrix_qsvt_m_controlled[8:,8:])
    #b = qsvt_instance.b_block_encoded_normalized
    b = qsvt_instance.b
    b = np.vstack((b, np.zeros_like(b),np.zeros_like(b),np.zeros_like(b)))
    b /= np.linalg.norm(b)
    q_x = matrix_qsvt_m_controlled @ b
    q_x = q_x[:4]
    q_x /= np.linalg.norm(q_x)
    print(q_x)


def qsvt():
    qml.PCPhase(phi_qsvt[0], dim=4, wires=[1,2,3])
    for i in range(1,len(projectors2)):
        qml.BlockEncode(qsvt_instance.A_hermitian, wires=[1,2,3])
        qml.PCPhase(phi_qsvt[i], dim=4, wires=[1,2,3])
    
@qml.qnode(qml.device("default.qubit", wires=[0,1,2,3]))
def qsvt_run():
    qml.StatePrep(qsvt_instance.b.T/np.linalg.norm(qsvt_instance.b), wires=[2, 3])
#    qml.Identity(wires=[0,1,2,3])#0is ancilla qubit
    qml.Hadamard(wires=[0])
    qml.ctrl(qsvt, control=(0,), control_values=(0,))()
    qml.ctrl(qml.adjoint(qsvt), control=(0,), control_values=(1,))()

    qml.Hadamard(wires=[0])

    return qml.state()

#print(qml.draw(qsvt_run,level='device', show_all_wires=True)())

qsvt_state = qsvt_run()
qsvt_result = qsvt_state[0][:4]
qsvt_result /= np.linalg.norm(qsvt_result)
with np.printoptions(precision=3, linewidth=200):
    print(qsvt_result.T)

# qsvt_instance.construct_string_qsvt_complete()
# qsvt_instance.write_and_import_kernel_qsvt_complete()

# qsvt_instance.sample()
# print('Samples global:', qsvt_instance.samples)

# qsvt_instance.create_samples_dict_ordered_be_and_reduced_b_be()
# print('Samples b register:', qsvt_instance.samples_dict_ordered_reduced_b_be)

# #print()
# #print(qsvt_instance.get_state())
# #qsvt_instance.create_quantum_state_amplitudes_dict_ordered_be()
# #print(qsvt_instance.state_amplitudes_dict_ordered_be)

# print(qsvt_instance.bit_strings_big_endian_all)
# print(qsvt_instance.bit_strings_big_endian_qvector_b)
# print(qsvt_instance.samples_dict_ordered_be)
# print(qsvt_instance.samples_dict_ordered_reduced_b_be)