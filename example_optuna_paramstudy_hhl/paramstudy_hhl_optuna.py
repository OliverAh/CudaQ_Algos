import cudaq
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
import itertools
import optuna
import argparse
from typing import List

import sys
import os
import time
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from src import hhl





#for t in cudaq.get_targets():
#    print(t.name, t.description, t.num_qpus())
#    print()
#sys.exit()
def target_hhl(kwargs, qpu_id):#, results):
    #print('kwargs:', kwargs)
    #print('qpu_id:', qpu_id)
    #print('results:', results)
    
    hhl_instance = hhl.HHL(**kwargs,
                            cudaq_target = 'nvidia',
                            cudaq_target_option = 'mqpu,fp64',
                            verbose = 0)
    
    hhl_instance.construct_string_hhl_complete()
    hhl_instance.write_and_import_kernel_hhl_complete(remove_file_after_import=True)
    
    #print(qpu_id)
    hhl_instance.sample_async(shots_count=int(1e6), qpu_id=qpu_id)
    #hhl_instance.get_state_async(qpu_id=qpu_id)
    hhl_instance.samples = hhl_instance.samples.get()
    #hhl_instance.quantum_state = hhl_instance.quantum_state.get()
    
    hhl_instance.create_samples_dict_ordered_be_and_reduced_b_be()
    #hhl_instance.create_quantum_state_amplitudes_dict_ordered_be()

    samples_count = np.array(list(hhl_instance.samples_dict_ordered_reduced_b_be.values()))
    samples_count_ratio_1_0 = samples_count[1]/samples_count[0]
    w_ratio_1_0 = hhl_instance.classical_solution[1]/hhl_instance.classical_solution[0]
    ratio_max_min_samples_count = np.max(samples_count)/np.min(samples_count)
    ratio_max_min_w = np.max(hhl_instance.classical_solution)/np.min(hhl_instance.classical_solution)
    
    obj_ratio_1_0 = (samples_count_ratio_1_0 - w_ratio_1_0)**2
    obj_ratio_max_min_samples_count = (ratio_max_min_samples_count - ratio_max_min_w)**2

    #results[qpu_id] = (obj_ratio_1_0, obj_ratio_max_min_samples_count)

    return obj_ratio_1_0, obj_ratio_max_min_samples_count


def objective(trial:optuna.Trial, list_gpu_ids_to_use:List):
    #system_size = trial.suggest_int('system_size', 4, 8, step=4, log=False)
    system_size = 16
    qvector_clock_size = trial.suggest_int('qvector_clock_size', 4, 26, step=1, log=False)
    t_hamiltonian_simulation = trial.suggest_float('t_hamiltonian_simulation', 0.5, 10.0, step=0.1, log=False)
    r_hamiltonian_simulation = trial.suggest_int('r_hamiltonian_simulation', 1, 32, step=1, log=False)

    qpu_id = trial.number % len(list_gpu_ids_to_use)
    
    #results = [None]*len(list_gpu_ids_to_use)

    kwargs = {'system_size': system_size,
              'qvector_clock_size': qvector_clock_size,
              't_hamiltonian_simulation': [t_hamiltonian_simulation],
              'r_hamiltonian_simulation': r_hamiltonian_simulation}
    obj_ratio_1_0, obj_ratio_max_min_samples_count = target_hhl(kwargs, qpu_id)#, results)
    
    return obj_ratio_1_0, obj_ratio_max_min_samples_count


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Optuna parameter study wrapping cudaq implementation of HHL algorithm')
    parser.add_argument('--n_trials', type=int, help='Number of trials to run per process')
    parser.add_argument('--extract_trials_to_csv', type=bool, default=False, help='Read study.get_trials() and write result to csv file')
    #parser.add_argument('--csv_filenamepath', type=str, default='./tmp/paramstudy_hhl.csv', help='Filenamepath to write csv file')

    parsed_args = parser.parse_args()
    print(parsed_args)
    print(parsed_args.n_trials)


    list_gpu_ids_to_use = [0,1,2,3]
    storage_filenamepath = './tmp/optuna_journal_storage_poisson_16x16.log'
    csv_filenamepath = pathlib.Path('tmp','paramstudy_hhl_poisson_16x16.csv')
    study_name = 'paramstudy_hhl_poisson_16x16'

    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(storage_filenamepath))
    #storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(storage=storage, study_name=study_name,
                                 directions=('minimize', 'minimize'),
                                 load_if_exists=True)
    if not parsed_args.extract_trials_to_csv:
        study.optimize(lambda trial: objective(trial, list_gpu_ids_to_use),
                   n_trials=parsed_args.n_trials, n_jobs=1)
    else:
        trials_df = study.trials_dataframe()
        trials_df.to_csv(csv_filenamepath)
        print('Best trials:')
        for i, frozentrial in enumerate(study.best_trials):
            print('', 'Best trial:', i)
            for key, val in frozentrial.__dict__.items():
                print('', '|', key, ':', val)
    
    #print(study.get_trials())
    #print(study.trials_dataframe())
    #trials_df = study.trials_dataframe()
    #trials_df.to_csv('tmp/paramstudy_hhl_poisson_2x2.csv')