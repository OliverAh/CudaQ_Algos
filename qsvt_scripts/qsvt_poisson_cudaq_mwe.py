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

#

# A = np.array([
#        [0.65713691, -0.05349524, 0.08024556, -0.07242864],
#        [-0.05349524, 0.65713691, -0.07242864, 0.08024556],
#        [0.08024556, -0.07242864, 0.65713691, -0.05349524],
#        [-0.07242864, 0.08024556, -0.05349524, 0.65713691],
#    ])

# A_Poisson = np.array([
#         [2., -1., 0., 0.],
#         [-1., 2., -1., 0.],
#         [0., -1., 2., -1.],
#         [0., 0., -1., 2.]
#     ])
#
# b = np.ones(A_Poisson.shape[0]) # abitrary stateprep currently not implemented, but can be provided by user



# qsvt_instance = qsvt.QSVT(A = A,
#                          b = np.ones(A.shape[0]),
qsvt_instance = qsvt.QSVT(
                        system_size=4,# must currently be a power of 2, implementation of padding will follow
                        cudaq_target = 'nvidia',#default single gpu simulator
                        #cudaq_target = 'qpp-cpu',#cpu simulator
                        cudaq_target_option = 'fp64',
                        verbose=99)

##########
# 
# Compute angles for Projector-Controlled-Phaseshifts
#
##########
cond = qsvt_instance.compute_condition_number()
print('Condition number:', cond)

tic = time.time()
kappa = cond
poly_oneoverx, scale_oneoverx = pyqsp.poly.PolyOneOverX().generate(kappa=kappa, return_coef=True, ensure_bounded=True, return_scale=True)
toc = time.time()
if qsvt_instance.verbose > 2:
    print('##### \n# Computed poly_oneoverx in', f'{toc-tic}s \n#####')

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
# Construct and run Pennylane simulation (optional)
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


##########
# 
# Construct Cudaq kernel and write to file
#
##########
qsvt_instance.construct_string_qsvt_complete()

qsvt_instance.write_kernel_qsvt_complete()


##########
# 
# Import Cudaq kernel from file and compile it
#
##########
qsvt_instance.import_kernel_qsvt_complete(remove_file_after_import=False)#, filepath='tmp', filename='kernel_qsvt_complete_from_class_c9f5e4c5_1327_448d_9983_d784994ca5e4.py')
qsvt_instance.compile_kernel_qsvt_complete()

##########
# 
# Draw Cudaq circuit (optional)
#
##########
#qsvt_instance.draw()
#print(qsvt_instance.circuit_string)

##########
# 
# Sample and/or get state vector for solution and postprocess results
# Postprocessing is about to be moved tot the QSVT class, so the user does not have to do it manually
# For samples the postprocessing here is an artifact from debugging and can probably be simplified
#
##########

bit_strings_of_interest = qsvt_instance.bit_strings_big_endian_qvector_b

##
# Samples
##
samples = qsvt_instance.sample(shots_count=int(1e6))
tic = time.time()
samples_dict = {key: val/qsvt_instance.samples_shots_count for key, val in samples.items()}
samples_dict = {key: val for key, val in samples_dict.items() if key in bit_strings_of_interest}
samples_dict = {key: np.sqrt(val) for key, val in samples_dict.items()}
samples_dict = {key: val/np.linalg.norm(list(samples_dict.values())) for key, val in samples_dict.items()}
print('Samples:\n', {key: samples_dict[key] for key in bit_strings_of_interest if key in samples_dict.keys()})
samples.clear()
toc = time.time()
if qsvt_instance.verbose > 2:
    print('##### \n# Finished postprocessing cudaq samples in', f'{toc-tic}s \n#####')

##
# State vector
##
state_obj = qsvt_instance.get_state()
with np.printoptions(precision=3, linewidth=200):
    state = state_obj
    print('Quantum state:\n', state)

    state2 = state_obj.amplitudes(qsvt_instance.bit_strings_big_endian_qvector_b)
    state2 /= np.linalg.norm(state2)
    print('Quantum solution:\n', state2/np.linalg.norm(state2))
    
##
# Classical solution
##
print()
print('Classical solution:')
print(qsvt_instance.classical_solution.T/np.linalg.norm(qsvt_instance.classical_solution))
print()
print('Bitstrings of interest:')
print(qsvt_instance.bit_strings_big_endian_qvector_b)
