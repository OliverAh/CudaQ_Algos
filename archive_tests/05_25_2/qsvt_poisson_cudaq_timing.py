##################################################################
###
### This is intended to be executed in the apptainer container built using ./container/build_apptainer_sif.sh
### All necessary packages are installed in "base" environment of the container, so no activation of an environment is needed.
###
### The container is based on nvidias cudaq container and then adds some pip packages such as pennylane, pyqsp, etc.
###
##################################################################


print('Start imports')
import cudaq
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
import matplotlib.pyplot as plt

import numpy as np
import pyqsp
import pyqsp.angle_sequence
import pennylane as qml

from src import qsvt
import time

print('Finished imports')

A = np.array([
       [0.65713691, -0.05349524, 0.08024556, -0.07242864],
       [-0.05349524, 0.65713691, -0.07242864, 0.08024556],
       [0.08024556, -0.07242864, 0.65713691, -0.05349524],
       [-0.07242864, 0.08024556, -0.05349524, 0.65713691],
   ])
b = np.array([1., 2., 3., 4.]).reshape((4,1))

A_Poisson = np.array([
        [2., -1., 0., 0.],
        [-1., 2., -1., 0.],
        [0., -1., 2., -1.],
        [0., 0., -1., 2.]
    ])
b_Poisson = np.array([1., 1., 1., 1.]).reshape((4,1))


#qsvt_instance = qsvt.QSVT(A = A,
#                          b = b,
qsvt_instance = qsvt.QSVT(
                        system_size=4,
                        cudaq_target = 'nvidia',
                        cudaq_target_option = 'fp64',
                        verbose=99)
#qsvt_instance.BlockEncode_A_unitary()
cond = qsvt_instance.compute_condition_number()
print('Condition number:', cond)
#with np.printoptions(precision=2, linewidth=200):
#    print(qsvt_instance.A_block_encoded_unitary)

tic = time.time()
kappa = cond
poly_oneoverx, scale_oneoverx = pyqsp.poly.PolyOneOverX().generate(kappa=kappa, return_coef=True, ensure_bounded=True, return_scale=True)
toc = time.time()
if qsvt_instance.verbose > 2:
    print('##### \n# Computed poly_oneoverx in', f'{toc-tic}s \n#####')
#print(poly_oneoverx)
#print(scale_oneoverx)

tic = time.time()
angles_poly_oneoverx = pyqsp.angle_sequence.QuantumSignalProcessingPhases(poly_oneoverx, signal_operator="Wx", tolerance=0.00001)
toc = time.time()
toc = time.time()
if qsvt_instance.verbose > 2:
    print('##### \n# Computed angles in', f'{toc-tic}s \n#####')
tic = time.time()
phi_qsvt = qml.transform_angles(angles_poly_oneoverx, "QSP", "QSVT")
toc = time.time()
if qsvt_instance.verbose > 2:
    print('##### \n# Converted angles in', f'{toc-tic}s \n#####')

qsvt_instance.angles_poly_oneoverx = phi_qsvt
#qsvt_instance.angles_poly_oneoverx = [phi_qsvt[0], phi_qsvt[1]]

##########
# 
# Construct the QSVT circuit
#
##########
tic = time.time()
qsvt_instance.construct_qsvt_circuit_pennylane()
toc = time.time()
if qsvt_instance.verbose > 2:
    print('##### \n# Created pennylane circuit in', f'{toc-tic}s \n#####')
#print(qml.draw(qsvt_instance.circuit_pennylane, decimals=2, show_all_wires=True)())
tic = time.time()
qsvt_state_internal = qsvt_instance.circuit_pennylane()
toc = time.time()
if qsvt_instance.verbose > 2:
            print('##### \n# Pennylane finished sampling in', f'{toc-tic}s \n#####')
with np.printoptions(precision=3, linewidth=200):
    print(qsvt_state_internal)
if qsvt_state_internal.shape[0] == 1:
    qsvt_state_internal = qsvt_state_internal[0][:qsvt_instance.system_size]
else:
    qsvt_state_internal = qsvt_state_internal[:qsvt_instance.system_size]
qsvt_state_internal /= np.linalg.norm(qsvt_state_internal)
with np.printoptions(precision=3, linewidth=200):
    print(qsvt_state_internal.T)

#print(qml.draw(qsvt_instance.circuit_pennylane, show_all_wires=True)())   
#print(qml.draw(qsvt_instance.circuit_pennylane, decimals=2, show_all_wires=True)())
qsvt_instance.construct_string_qsvt_complete()

qsvt_instance.write_kernel_qsvt_complete()


##########
# 
# Execute the QSVT circuit
#
##########


qsvt_instance.import_kernel_qsvt_complete(remove_file_after_import=False)#, filepath='tmp', filename='kernel_qsvt_complete_from_class_c9f5e4c5_1327_448d_9983_d784994ca5e4.py')
qsvt_instance.compile_kernel_qsvt_complete()

#qsvt_instance.draw()
#print(qsvt_instance.circuit_string)

bit_strings_of_interest = ['0000', '0010', '0001','0011']
#bit_strings_of_interest = ['0000', '0100', '0010','0110']
#bit_strings_of_interest = ['0000', '1000', '0100','1100']
bit_strings_of_interest = ['00000', '00001', '00010','00011','00100','00101','00110','00111']

samples = qsvt_instance.sample(shots_count=int(1e6))
tic = time.time()
samples_dict = {key: val/qsvt_instance.samples_shots_count for key, val in samples.items()}
samples_dict = {key: val for key, val in samples_dict.items() if key in bit_strings_of_interest}
samples_dict = {key: np.sqrt(val) for key, val in samples_dict.items()}
samples_dict = {key: val/np.linalg.norm(list(samples_dict.values())) for key, val in samples_dict.items()}
print('Samples:', {key: samples_dict[key] for key in bit_strings_of_interest if key in samples_dict.keys()})
samples.clear()
toc = time.time()
if qsvt_instance.verbose > 2:
    print('##### \n# Finished postprocessing cudaq samples in', f'{toc-tic}s \n#####')
#cudaq.reset_target()

#print(qsvt_instance.bit_strings_big_endian_all)
state_obj = qsvt_instance.get_state()
#print('State:', state)
with np.printoptions(precision=3, linewidth=200):
    # state = state_obj.amplitudes(['0000', '0010', '0001','0011']) #bit_strings_of_interest)
    # state /= np.linalg.norm(state)
    # print('State:\n', state/np.linalg.norm(state))
    # sol = qsvt_instance.A @ state
    # sol /= np.linalg.norm(sol)
    # print(sol)

    # state2 = state_obj.amplitudes(['0000', '0100', '0010', '0110'])
    # state2 /= np.linalg.norm(state2)
    # print('State:\n', state2/np.linalg.norm(state2))
    # sol = qsvt_instance.A @ state2
    # sol /= np.linalg.norm(sol)
    # print(sol)

    # state3 = state_obj.amplitudes(['0000', '1000', '0100', '1100'])
    # state3 /= np.linalg.norm(state3)
    # print('State:\n', state3/np.linalg.norm(state3))
    # sol = qsvt_instance.A @ state3
    # sol /= np.linalg.norm(sol)
    # print(sol)

    # state4 = state_obj.amplitudes(['0000', '0001', '0010','0011']) #bit_strings_of_interest)
    # state4 /= np.linalg.norm(state4)
    # print('State:\n', state4/np.linalg.norm(state4))
    # sol = qsvt_instance.A @ state4
    # sol /= np.linalg.norm(sol)
    # print(sol)

    state4 = state_obj
    print('State:\n', state4)

    state5 = state_obj.amplitudes(qsvt_instance.bit_strings_big_endian_qvector_b) #bit_strings_of_interest)
    state5 /= np.linalg.norm(state5)
    print('State:\n', state5/np.linalg.norm(state5))
    sol = qsvt_instance.A @ state5
    sol /= np.linalg.norm(sol)
    print(sol)


print()
print('Classical solution:')
print(qsvt_instance.classical_solution.T/np.linalg.norm(qsvt_instance.classical_solution))
print()
print(qsvt_instance.bit_strings_big_endian_qvector_b)
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