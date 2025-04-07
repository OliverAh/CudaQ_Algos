import pandas as pd
import matplotlib.pyplot as plt

system_zize = '16x16'
df = pd.read_csv(f'tmp/paramstudy_hhl_poisson_{system_zize}.csv')

df_sorted = df.sort_values(by=['params_qvector_clock_size', 'params_r_hamiltonian_simulation', 'params_t_hamiltonian_simulation'])
df_sorted = df.sort_values(by=['values_0', 'values_1'])
df_of_interest = df_sorted[['values_0', 'values_1', 'params_qvector_clock_size', 'params_r_hamiltonian_simulation', 'params_t_hamiltonian_simulation']]
with pd.option_context('display.max_rows', None):
    print(df_of_interest)
    #print(df_sorted[['values_0']])
    print(df_of_interest.cov())


sxsy = (('values_0', 'values_1'),
        ('params_qvector_clock_size', 'values_1'),
        ('params_r_hamiltonian_simulation', 'values_1'),
        ('params_t_hamiltonian_simulation', 'values_1'))
for (scatter_x, scatter_y) in sxsy:
    df_sorted.plot.scatter(x=scatter_x, y=scatter_y, s=50)
    plt.savefig(f'tmp/poisson_{system_zize}_{scatter_x}_{scatter_y}.png')
    plt.close()
    df_sorted.plot.line(x=scatter_x, y=scatter_y)
    plt.savefig(f'tmp/poisson_{system_zize}_{scatter_x}_{scatter_y}_line.png')
    plt.close()