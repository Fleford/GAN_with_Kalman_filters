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

# Z is the observed heads through time
Z = np.loadtxt('ts_ref_rand.txt')
Z=Z.reshape(Z.shape[0], 1)

# Prepare k_array
k_array = np.random.rand(100, nrow, ncol)
plt.matshow(k_array[0])
plt.show()
print("k_array")
print(k_array)
print(k_array.shape)

# def GSLIB2ndarray(data_file, kcol, nz, nx, ny):
#     Karray = np.ndarray(shape=(nz, ny, nx), dtype=float, order='F')
#
#     if ny > 1:
#         array_k = np.ndarray(shape=(nz, ny, nx), dtype=float, order='F')
#     else:
#         array_k = np.zeros(nx)
#
#     with open(data_file) as myfile:  # read first two lines
#         head = [next(myfile) for x in range(2)]
#         line2 = head[1].split()
#         ncol = int(line2[0])  # get the number of columns
#         for icol in range(0, ncol):  # read over the column names
#             head = [next(myfile) for x in range(1)]
#             if icol == kcol:
#                 col_name = head[0].split()[0]
#         for iz in range(0, nz):
#             for iy in range(0, ny):
#                 for ix in range(0, nx):
#                     head = [next(myfile) for x in range(1)]
#                     Karray[iz][ny - 1 - iy][ix] = head[0].split()[kcol]
#                     array_k[iz][ny - 1 - iy][ix] = Karray[iz][ny - 1 - iy][ix]
#                     # array_k[iz][ny - 1 - iy][ix] = math.exp(Karray[iz][ny - 1 - iy][ix])
#     return array_k, col_name
#
#
# k_array, m = GSLIB2ndarray('sgsim.out', 0, nr, ncol, nrow)

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
for pp in range(0, maxIter):
    for i in range(0,nr):   
        print("Running MODFLOW for k realization " + str(i))
        lpf = flopy.modflow.ModflowLpf(mf, hk=np.exp(k_array[i,:,:]), vka=vka, sy=sy, ss=ss, laytyp=laytyp, ipakcb=53)
        mf.write_input()                               
        success, mfoutput = mf.run_model(silent=False, pause=False)
        if not success:
            raise Exception('MODFLOW did not terminate normally.')
        headobj = bf.HeadFile(modelname+'.hds')
        ts = headobj.get_ts(idx)    # Extract transient heads
        # print("ts")
        # print(ts)
        # print(ts.shape)
        tss[:,0] = ts[:,0]
        tss[:,i+1] = ts[:,1]

    # Convert k realizations into a 2d matrix with each column representing a realization
    yf=k_array.reshape(nr, nrow * ncol)
    yf=yf.transpose()

    print("k_array")
    print(k_array)
    print(k_array.shape)

    print("yf")
    print(yf)
    print(yf.shape)

    # Calculate the mean k value for a given location in the domain
    ym = np.array(yf.mean(axis=1))    # Mean of the y_f (the k_array)
    ym=ym.reshape(ym.shape[0],1)    
    dmf=yf-ym

    # The simulated drawdown heads using k_array
    # Each column is realization
    df=tss[:, 1:]
    dm = np.array(df.mean(axis=1)) 
    dm = dm.reshape(dm.shape[0], 1)
    ddf = df-dm

    print("df")
    print(df)
    print(df.shape)

    print("ddf")
    print(ddf)
    print(ddf.shape)
    
    Cmd_f = (np.dot(dmf, ddf.T))/(nr-1)  # The cross-covariance matrix between the k values and the field
    Cdd_f = (np.dot(ddf, ddf.T))/(nr-1)  # The auto covariance of predicted data (transient heads of k realizations)

    print("dmf")
    print(dmf)
    print(dmf.shape)

    print("Cmd_f")
    print(Cmd_f)
    print(Cmd_f.shape)

    print("Cdd_f")
    print(Cdd_f)
    print(Cdd_f.shape)

    CD = np.eye(101) * 0.01
    R = linalg.cholesky(CD,lower=True) #Matriz triangular inferior
    U = R.T   #Matriz R transpose
    # p, w = np.linalg.eig(CD)

    print("R")
    print(R)
    print(R.shape)

    print("U")  # Only the U variable is used, U = CD = Identity matrix times 0.01
    print(U)
    print(U.shape)

    # print("p")
    # print(p)
    # print(p.shape)
    #
    # print("w")
    # print(w)
    # print(w.shape)

    # Z is the observed heads through time (just a single vector)
    print("Z")
    print(Z)
    print(Z.shape)

    # Convert Z(true transient heads) into the same dimension as ddf(realization transient heads)
    aux = np.repeat(Z, nr, axis=1)

    # Create a noise matrix with same dimensions as ddf(realization transient heads)
    mean = 0 * (Z.T)
    noise = np.random.multivariate_normal(mean[0], np.eye(len(Z)), nr).T

    print("aux")
    print(aux)
    print(aux.shape)

    print("mean")
    print(mean)
    print(mean.shape)

    print("noise")
    print(noise)
    print(noise.shape)

    # Create a noisy version of aux(Z(true transient heads) with the dimensions of ddf(realization transient heads))
    # The noise is scaled with math.sqrt(alpha[pp]) * (identity_matrix * 0.01)
    d_obs = aux+math.sqrt(alpha[pp])*np.dot(U, noise)

    print("d_obs")
    print(d_obs)
    print(d_obs.shape)

    # Analysis step
    # varn = 0.99
    varn=1-1/math.pow(10, 2)

    # Reminder: Cdd_f is the auto covariance of predicted data (transient heads of k realizations)

    u, s, vh = linalg.svd(Cdd_f+alpha[pp]*CD)
    v = vh.T
    diagonal = s
    for i in range(len(diagonal)):
        # Only eigen vectors the contribute to 99% of solution
        if (sum(diagonal[0:i+1]))/(sum(diagonal)) > varn:
            diagonal = diagonal[0:i+1]
            break

    #     print("i in loop")
    #     print(i)
    # print("i out loop")
    # print(i)
    u=u[:,0:i+1]
    v=v[:,0:i+1]
    ess = np.diag(diagonal**(-1))
    K=np.dot(Cmd_f,(np.dot(np.dot(v, ess), (u.T))))   # K = [Cmd_f][v][ess][u.T]

    print("u")
    print(u)
    print(u.shape)

    print("v")
    print(v)
    print(v.shape)

    print("ess")
    print(ess)
    print(ess.shape)

    print("K")
    print(K)
    print(K.shape)

    # Reminder: yf is the 2d matrix of k realizations
    ya = yf + (np.dot(K, (d_obs - df))) # ya = yf + [K]([d_obs - ])
    ya = ya.transpose()
    k_array=ya.reshape(nr, ncol, nrow)
   
    plt.figure(pp)
    ttl = 'figure 11'.format(idx[0] + 1, idx[1] + 1, idx[2] + 1)
    plt.title(ttl)
    plt.xlabel('time')
    plt.ylabel('head')
    for i in range(1,nr+1):  
        plt.plot(tss[:, 0], tss[:, i], 'b')
        plt.plot(Z, 'r-')
        plt.savefig('tutorial2-ts' + str(pp)+ '.png')

    # Plot first k array
    plt.matshow(k_array[0])
    plt.savefig('tutorial2-k' + str(pp) + '.png')
    plt.show()

    # # Remove when done
    # break





# Perturb the vector of observations


# Create the headfile and budget file objects
#headobj = bf.HeadFile(modelname+'.hds')
#times = headobj.get_times()
#cbb = bf.CellBudgetFile(modelname+'.cbc')

# Plot the head versus time
#idx = (0, int(nrow/2) - 1, int(ncol/2) - 1)
#idx = (0, 20, 20)
#ts = headobj.get_ts(idx)
