# Import relevant modules and methods.
import numpy as np
import pyqsp
from pyqsp import angle_sequence, response
from pyqsp.poly import PolyOneOverX

import sys
print(sys.argv)
kappa = None
if sys.argv[1] == '--kappa':
    kappa_str = str(sys.argv[2])
    kappa = int(sys.argv[2])

poly_oneoverx, sclae_poly_oneoverx = PolyOneOverX().generate(kappa=kappa, return_coef=True, ensure_bounded=True, return_scale=True)
tru_func_oneoverx = lambda x: sclae_poly_oneoverx * 1.0 / (x+1e-12)

angles_poly_oneoverx = angle_sequence.QuantumSignalProcessingPhases(poly_oneoverx, signal_operator="Wx")

response.PlotQSPResponse(
                         angles_poly_oneoverx, 
                         pcoefs = poly_oneoverx,
                         target = tru_func_oneoverx,
                         sym_qsp = False,
                         simul_error_plot = False
    )

if kappa < 1000. and kappa >= 100.:
    kappa_str = ''.join(['0',kappa_str])
elif kappa < 100. and kappa >= 10.:
    kappa_str = ''.join(['00', kappa_str])
elif kappa < 10.:
    kappa_str = ''.join(['000',kappa_str])
    
response.plt.savefig(f'plots/kappa_{kappa_str}.png', dpi=300)

angles = np.array(angles_poly_oneoverx)
print(angles)
print(poly_oneoverx)
np.save('angles/kappa_{}_angles_{}.npy'.format(kappa_str, format(angles.shape[0], '0' + str(7) + 'd')), angles)