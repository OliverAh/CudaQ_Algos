#import cudaq
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent))
import matplotlib.pyplot as plt

import numpy as np
import pyqsp
import pyqsp.angle_sequence
import pennylane as qml


from src import qsvt_pennylane as qsvt



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
#qsvt_instance.BlockEncode_A_unitary()
cond = qsvt_instance.compute_condition_number()
print('Condition number:', cond)
#with np.printoptions(precision=2, linewidth=200):
#    print(qsvt_instance.A_block_encoded_unitary)


kappa = cond
poly_oneoverx, scale_oneoverx = pyqsp.poly.PolyOneOverX().generate(kappa=kappa, return_coef=True, ensure_bounded=True, return_scale=True)
#print(poly_oneoverx)
#print(scale_oneoverx)

angles_poly_oneoverx = pyqsp.angle_sequence.QuantumSignalProcessingPhases(poly_oneoverx, signal_operator="Wx", tolerance=0.00001)
phi_qsvt = qml.transform_angles(angles_poly_oneoverx, "QSP", "QSVT")

qsvt_instance.angles_poly_oneoverx = phi_qsvt



qsvt_instance.construct_qsvt_circuit_pennylane()
#print(qml.draw(qsvt_instance.circuit_pennylane, decimals=2, show_all_wires=True)())
qsvt_state_internal = qsvt_instance.circuit_pennylane()
qsvt_state_internal = qsvt_state_internal[0][:4]
qsvt_state_internal /= np.linalg.norm(qsvt_state_internal)
with np.printoptions(precision=3, linewidth=200):
    print(qsvt_state_internal.T)
    


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