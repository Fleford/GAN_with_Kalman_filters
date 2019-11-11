import numpy as np
import flopy
import math
import matplotlib.pyplot as plt
#import pandas as pd
from scipy import array, linalg, dot

import flopy.utils.binaryfile as bf
# Model domain and grid definition
Lx = 1000.
Ly = 1000.
ztop = 10.
zbot = -50.
nlay = 1
nrow = 50  # Changed from 50 to 97
ncol = 50  # Changed from 50 to 97
delr = Lx / ncol
delc = Ly / nrow
delv = (ztop - zbot) / nlay
botm = np.linspace(ztop, zbot, nlay + 1)
hk = 1.
vka = 1.
sy = 0.1
ss = 1.e-4
laytyp = 1

nr = 100  # Number of realizations

maxIter = 4
alpha = np.zeros(maxIter)
for pp in range(maxIter):
    alpha[pp] = (2**(maxIter-pp))
# print("alpha")
# print(alpha)
# print(alpha.shape)

# Create and save a k_array
k_array = np.random.rand(nrow, ncol)
np.savetxt("k_array_ref_rand.txt", k_array)
plt.matshow(k_array)
plt.savefig('k_array_ref_rand.png')

# Prepare k_array
k_array = np.loadtxt("k_array_ref_rand.txt")
plt.matshow(k_array)
plt.show()
print("k_array")
print(k_array)
print(k_array.shape)

# Variables for the BAS package
# Note that changes from the previous tutorial!
ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
strt = 10. * np.ones((nlay, nrow, ncol), dtype=np.float32)

# Time step parameters
nper = 2
perlen = [1, 100]
nstp = [1, 100]
steady = [True, False]


# Flopy objects
modelname = 'tutorial2'
mf = flopy.modflow.Modflow(modelname, exe_name='mf2005')
dis = flopy.modflow.ModflowDis(mf, nlay, nrow, ncol, delr=delr, delc=delc,
                               top=ztop, botm=botm[1:],
                               nper=nper, perlen=perlen, nstp=nstp,
                               steady=steady)

ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
ibound[:, :, 0] = -1
ibound[:, :, -1] = -1
strt = np.ones((nlay, nrow, ncol), dtype=np.float32)
strt[:, :, 0] = 10.
strt[:, :, -1] = 0.
bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)

pcg = flopy.modflow.ModflowPcg(mf)

# Create the flopy ghb object
#ghb = flopy.modflow.ModflowGhb(mf, stress_period_data=stress_period_data)

# Create the well package
# Remember to use zero-based layer, row, column indices!
pumping_rate = -500.
wel_sp1 = [[0, nrow/2 - 1, ncol/2 - 1, 0.]]
wel_sp2 = [[0, nrow/2 - 1, ncol/2 - 1, -500.]]
#wel_sp3 = [[0, nrow/2 - 1, ncol/2 - 1, pumping_rate]]
stress_period_data = {0: wel_sp1, 1: wel_sp2}
wel = flopy.modflow.ModflowWel(mf, stress_period_data=stress_period_data)

# Output control
stress_period_data = {}
for kper in range(nper):
    for kstp in range(nstp[kper]):
        stress_period_data[(kper, kstp)] = ['save head',
                                            'save drawdown',
                                            'save budget',
                                            'print head',
                                            'print budget']
oc = flopy.modflow.ModflowOc(mf, stress_period_data=stress_period_data,  
                             compact=True)

# Write the model input files

tss = np.ones((101,nr+1), dtype=np.float32)
idx = (0, int(nrow/2) - 1, int(ncol/2) - 1)


# Run the model
print("Running MODFLOW for k reference")
lpf = flopy.modflow.ModflowLpf(mf, hk=np.exp(k_array), vka=vka, sy=sy, ss=ss, laytyp=laytyp, ipakcb=53)
mf.write_input()
success, mfoutput = mf.run_model(silent=False, pause=False)
if not success:
    raise Exception('MODFLOW did not terminate normally.')
headobj = bf.HeadFile(modelname+'.hds')
ts = headobj.get_ts(idx)    # Extract transient heads
np.savetxt("ts_ref_rand.txt", ts[:, 1])


