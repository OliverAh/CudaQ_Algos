import cudaq
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent))

from src import hhl

hhl_instance = hhl.HHL(t_hamiltonian_simulation=[1.0],
                        r_hamiltonian_simulation=2,
                        cudaq_target = 'nvidia',
                        cudaq_target_option = 'fp64',
                        verbose=99)
hhl_instance.construct_string_hhl_complete()
hhl_instance.write_and_import_kernel_hhl_complete()

hhl_instance.sample()
print('Samples global:', hhl_instance.samples)

hhl_instance.create_samples_dict_ordered_be_and_reduced_b_be()
print('Samples b register:', hhl_instance.samples_dict_ordered_reduced_b_be)

#print()
#print(hhl_instance.get_state())
#hhl_instance.create_quantum_state_amplitudes_dict_ordered_be()
#print(hhl_instance.state_amplitudes_dict_ordered_be)