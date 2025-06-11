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

print('Finished imports\n')

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
                        system_size=8,
                        cudaq_target = 'nvidia',
                        #cudaq_target = 'qpp-cpu',
                        cudaq_target_option = 'fp64',
                        verbose=99)
#qsvt_instance.BlockEncode_A_unitary()
cond = qsvt_instance.compute_condition_number()
#print('Condition number:', cond)

#angles_internal, scale_oneoverx = qsvt.PolynomialsAngles_Calculater().calculate_angles_oneoverx_default(kappa=cond, verbose=99)
#qsvt_instance.angles_poly_oneoverx = angles_internal
#qsvt_instance.angles_poly_oneoverx = [angles_internal[0], angles_internal[1]]

#f = 'qsp_angles/angles/kappa_0010_angles_0000920.npy'
#angles_loaded_file = qsvt.PolynomialsAngles_Loader().load_angles_from_npy_file(filenamepath=f)
#qsvt_instance.angles_poly_oneoverx = angles_loaded_file
#qsvt_instance.angles_poly_oneoverx = [angles_loaded_file[0], angles_loaded_file[1]]

f = 'qsp_angles/angles'
angles_loaded_dir = qsvt.PolynomialsAngles_Loader().load_suitable_angles_from_dir(angles_dir=f, kappa=cond, verbose=99)
#assert np.allclose(angles_loaded_file, angles_loaded_dir, atol=1e-5), "Angles do not match!"
qsvt_instance.angles_poly_oneoverx = angles_loaded_dir
#qsvt_instance.angles_poly_oneoverx = [angles_loaded_dir[0], angles_loaded_dir[1]]
#qsvt_instance.angles_poly_oneoverx = [0.0, angles_loaded_dir[1]]

##########
# 
# Construct the QSVT circuit
#
##########
tic = time.time()
qsvt_instance.construct_qsvt_circuit_pennylane(pennylane_device='lightning.qubit')
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
    print('Pennylane state complete:\n', qsvt_state_internal)
if qsvt_state_internal.shape[0] == 1:
    qsvt_state_internal = qsvt_state_internal[0][:qsvt_instance.system_size]
else:
    qsvt_state_internal = qsvt_state_internal[:qsvt_instance.system_size]
qsvt_state_internal /= np.linalg.norm(qsvt_state_internal)
with np.printoptions(precision=3, linewidth=200):
    print('Pennylane state b_vec normalized:\n', qsvt_state_internal.T, '\n')

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
bit_strings_of_interest = ['0000', '0001', '0010','0011']
#bit_strings_of_interest = ['0000', '0100', '0010','0110']
#bit_strings_of_interest = ['0000', '1000', '0100','1100']
#bit_strings_of_interest = ['00000', '00001', '00010','00011','00100','00101','00110','00111']
bit_strings_of_interest = qsvt_instance.bit_strings_big_endian_qvector_b

samples = qsvt_instance.sample(shots_count=int(1e6))
tic = time.time()
samples_dict = {key: val/qsvt_instance.samples_shots_count for key, val in samples.items()}
samples_dict = {key: val for key, val in samples_dict.items() if key in bit_strings_of_interest}
samples_dict = {key: np.sqrt(val) for key, val in samples_dict.items()}
samples_dict = {key: val/np.linalg.norm(list(samples_dict.values())) for key, val in samples_dict.items()}
print('Cudaq samples b_reg normalized:\n', {key: samples_dict[key] for key in bit_strings_of_interest if key in samples_dict.keys()})
samples.clear()
toc = time.time()
if qsvt_instance.verbose > 2:
    print('##### \n# Finished postprocessing cudaq samples in', f'{toc-tic}s \n#####')
#cudaq.reset_target()

state_obj = qsvt_instance.get_state()
with np.printoptions(precision=3, linewidth=200):
    state = state_obj
    print('Cudaq state complete:\n', state)

    state2 = state_obj.amplitudes(qsvt_instance.bit_strings_big_endian_qvector_b) #bit_strings_of_interest)
    state2 /= np.linalg.norm(state2)
    print('Cudaq state b_reg normalized:\n', state2, '\n')

    print('RHS for state, (not) rescaled:\n',
          qsvt_instance.A_unitary @ state2, '\n',
          qsvt_instance.A_unitary @ state2
          * qsvt_instance.A_block_encoded_unitary_scale
          * qsvt_instance.b_block_encoded_scale)

print()
print('Classical solution (not) normalized:')
print(qsvt_instance.classical_solution.T, '\n',
      qsvt_instance.classical_solution.T/np.linalg.norm(qsvt_instance.classical_solution))
print()
print('bit_strings_big_endian_qvector_b:')
print(qsvt_instance.bit_strings_big_endian_qvector_b)
