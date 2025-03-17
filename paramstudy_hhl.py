import cudaq
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
import itertools

import sys
import os
import time
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent))
from src import hhl

#for t in cudaq.get_targets():
#    print(t.name, t.description, t.num_qpus())
#    print()
#sys.exit()

# 'qvector_clock_size', 'system_size', 't_hamiltonian_simulation', 'r_hamiltonian_simulation')
params = {'qvector_clock_size': [4, 5, 6, 7, 8, 9, 10],
          'system_size': [4, 8],
          't_hamiltonian_simulation': [[t] for t in np.arange(0.5, 10.0, 0.1)],
          'r_hamiltonian_simulation': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])}
paramsets = pd.DataFrame(columns=params.keys())
for i,s in enumerate(itertools.product(*[params[key] for key in params.keys()])):
    paramsets.loc[i,:] = s
print(paramsets)
print(paramsets.index)

print('Num. GPUs available:', cudaq.num_available_gpus())
list_gpu_ids_to_use = [1,2,3]
print('List of GPU IDs to use:', list_gpu_ids_to_use)

hhls_dict = {}
print('Start all circuit runs')
for i in tqdm.tqdm(paramsets.index[:10]):
    hhls_dict[i] = {'hhl': hhl.HHL(qvector_clock_size=paramsets['qvector_clock_size'][i],
                            system_size=paramsets['system_size'][i],
                            t_hamiltonian_simulation=paramsets['t_hamiltonian_simulation'][i],
                            r_hamiltonian_simulation=paramsets['r_hamiltonian_simulation'][i],
                            cudaq_target = 'nvidia',
                            cudaq_target_option = 'mqpu,fp64',
                            verbose = 0),
                    'execution_started': False,
                    'execution_finished': False}
    
    hhls_dict[i]['hhl'].construct_string_hhl_complete()
    hhls_dict[i]['hhl'].write_and_import_kernel_hhl_complete(remove_file_after_import=True)
    qpu_id = i % len(list_gpu_ids_to_use)
    #print(list_gpu_ids_to_use[qpu_id])
    hhls_dict[i]['hhl'].sample_async(shots_count=int(1e6), qpu_id=list_gpu_ids_to_use[qpu_id])
    hhls_dict[i]['hhl'].get_state_async(qpu_id=list_gpu_ids_to_use[qpu_id])
    hhls_dict[i]['execution_started'] = True

print('Retrieve all results')
for id in tqdm.tqdm(hhls_dict.keys()):
    hhls_dict[id]['hhl'].samples = hhls_dict[id]['hhl'].samples.get()
    hhls_dict[id]['hhl'].quantum_state = hhls_dict[id]['hhl'].quantum_state.get()
    hhls_dict[id]['execution_finished'] = True

print('Create all ordered samples dicts')
for id in tqdm.tqdm(hhls_dict.keys()):
    hhls_dict[id]['hhl'].create_samples_dict_ordered_be_and_reduced_b_be()
    hhls_dict[id]['hhl'].create_quantum_state_amplitudes_dict_ordered_be()

print('Determine solution quality')
for id in tqdm.tqdm(hhls_dict.keys()):
    hhls_dict[id]['evaluation'] = {}
    samples_count = np.array(list(hhls_dict[id]['hhl'].samples_dict_ordered_reduced_b_be.values()))
    samples_count_ratio_1_0 = samples_count[1]/samples_count[0]
    w_ratio_1_0 = hhls_dict[id]['hhl'].classical_solution[1]/hhls_dict[id]['hhl'].classical_solution[0]
    ratio_max_min_samples_count = np.max(samples_count)/np.min(samples_count)
    ratio_max_min_w = np.max(hhls_dict[id]['hhl'].classical_solution)/np.min(hhls_dict[id]['hhl'].classical_solution)
    
    hhls_dict[id]['evaluation']['samples_count'] = samples_count
    hhls_dict[id]['evaluation']['samples_count_ratio_1_0'] = samples_count_ratio_1_0
    hhls_dict[id]['evaluation']['w_ratio_1_0'] = w_ratio_1_0
    hhls_dict[id]['evaluation']['ratio_max_min_samples_count'] = ratio_max_min_samples_count
    hhls_dict[id]['evaluation']['ratio_max_min_w'] = ratio_max_min_w
    
    #print('samples_count:', samples_count)
    #print('samples_count[1]/samples_count[0]:', samples_count[1]/samples_count[0])
    #print('w[1]/w[0]:', abcd.classical_solution[1]/abcd.classical_solution[0])
    #print('max(samples_count)/min(samples_count):', np.max(samples_count)/np.min(samples_count))
    #print('max(w)/min(w):', np.max(abcd.classical_solution)/np.min(abcd.classical_solution))
    #quantum_solution = samples_count / shots_count
    #print('samples normalized:', quantum_solution)
    #print('ratio Ax / b:', [(abcd.A_hermitian@quantum_solution)[i] / abcd.b_hermitian[i] for i in range(len(abcd.b_hermitian))])
    #print('Ax, b:', abcd.A_hermitian@quantum_solution, abcd.b_hermitian)
    #scaling_factor = np.average([abcd.b_hermitian[i] / (abcd.A_hermitian@quantum_solution)[i] for i in range(len(abcd.b_hermitian))])
    #print('scaling_factor:', scaling_factor)
    #quantum_solution *= scaling_factor
    #print(quantum_solution)


for id in hhls_dict.keys():
    print(hhls_dict[id]['evaluation']['samples_count_ratio_1_0'])