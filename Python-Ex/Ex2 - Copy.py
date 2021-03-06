import numpy as np
import flopy
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy import array, linalg, dot


# Model domain and grid definition
Lx = 1000.
Ly = 1000.
ztop = 10.
zbot = -50.
nlay = 1
nrow = 50
ncol = 50
delr = Lx / ncol
delc = Ly / nrow
delv = (ztop - zbot) / nlay
botm = np.linspace(ztop, zbot, nlay + 1)
hk = 1.
vka = 1.
sy = 0.1
ss = 1.e-4
laytyp = 1

nr=2

# Variables for the BAS package
# Note that changes from the previous tutorial!
ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
strt = 10. * np.ones((nlay, nrow, ncol), dtype=np.float32)

# Time step parameters
nper = 2
perlen = [1, 100]
nstp = [1, 100]
steady = [True, False]

def GSLIB2ndarray(data_file,kcol,nz,nx,ny):    
    Karray = np.ndarray(shape=(nz,ny,nx),dtype=float,order='F')
    
    if ny > 1:
        array_k = np.ndarray(shape=(nz,ny,nx),dtype=float,order='F')
    else:
        array_k = np.zeros(nx)
        
    with open(data_file) as myfile:   # read first two lines
        head = [next(myfile) for x in range(2)]
        line2 = head[1].split()
        ncol = int(line2[0])          # get the number of columns
        for icol in range(0, ncol):   # read over the column names
            head = [next(myfile) for x in range(1)]
            if icol == kcol:
                col_name = head[0].split()[0]
        for iz in range(0,nz):
            for iy in range(0,ny):
                for ix in range(0,nx):
                    head = [next(myfile) for x in range(1)]
                    Karray[iz][ny-1-iy][ix] = head[0].split()[kcol]
                    array_k[iz][ny-1-iy][ix] = math.exp(Karray[iz][ny-1-iy][ix])
    return array_k,col_name

k_array,m=GSLIB2ndarray('sgsim.out',0,nr,ncol,nrow)

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
import flopy.utils.binaryfile as bf

for i in range(0,nr):   
    lpf = flopy.modflow.ModflowLpf(mf, hk=k_array[i,:,:], vka=vka, sy=sy, ss=ss, laytyp=laytyp,ipakcb=53)
    mf.write_input()                               
    success, mfoutput = mf.run_model(silent=False, pause=False)
    if not success:
        raise Exception('MODFLOW did not terminate normally.')
    headobj = bf.HeadFile(modelname+'.hds')
    ts = headobj.get_ts(idx)
    tss[:,0] = ts[:,0]
    tss[:,i+1] = ts[:,1]
    
    

Z = np.loadtxt('Z.txt')
Z=Z.reshape(Z.shape[0],1)

yf=k_array.reshape(2, 2500)
yf=yf.transpose()
yf=np.log(yf)
ym = np.array(yf.mean(axis=1))    # Mean of the y_f
ym=ym.reshape(ym.shape[0],1)    
dmf=yf-ym

df=tss[:,1:]
dm = np.array(df.mean(axis=1)) 
dm = dm.reshape(dm.shape[0],1)   
ddf=df-dm

    
Cmd_f = (np.dot(dmf,ddf.T))/(nr-1);  # The cros-covariance matrix
Cdd_f = (np.dot(ddf,ddf.T))/(nr-1);  # The auto covariance of predicted data

CD=np.eye(101) * 0.01
R = linalg.cholesky(CD,lower=True) #Matriz triangular inferior
U = R.T   #Matriz R transposta
p , w =np.linalg.eig(CD)

aux = np.repeat(Z,nr,axis=1)
mean = 0*(Z.T)
noise=np.random.multivariate_normal(mean[0], np.eye(len(Z)), nr).T

d_obs = aux+math.sqrt(2)*np.dot(U,noise)  

 # Analysis step
varn=1-1/math.pow(10,2)

u, s, vh = linalg.svd(Cdd_f+2*CD); v = vh.T
diagonal = s
for i in range(len(diagonal)):
    if (sum(diagonal[0:i+1]))/(sum(diagonal)) > varn:
        diagonal = diagonal[0:i+1]
        break
    
u=u[:,0:i+1]
v=v[:,0:i+1]
ss = np.diag(diagonal**(-1))
K=np.dot(Cmd_f,(np.dot(np.dot(v,ss),(u.T))))

ya = yf + (np.dot(K,(d_obs-df)))
    
# Perturb the vector of observations


#aux = np.repeat(Z,nr,axis=1)
#mean = 0*(Z.T)

#noise=np.random.multivariate_normal(mean[0], np.eye(len(Z)), nr).T
#d_obs = aux+math.sqrt(alpha)*np.dot(U,noise)  
    
#k=tss[:,np.array([ False, True, True])]

#df=tss

# num_ens: number of ensemble size ->nr
# m_ens (yf): simulated head
# prod_ens (df): observation data

def ES_MDA(num_ens,m_ens,Z,prod_ens,alpha,CD,corr,numsave=2):
    varn=1-1/math.pow(10,numsave)
    # Initial Variavel 
    # Forecast step
    # Z: measurment
    # CD: covariance matrix of observed data measurement errors
    yf = m_ens                        # Non linear forward model 
    df = prod_ens                     # Observation Model
    numsave
    ym = np.array(yf.mean(axis=1))    # Mean of the y_f
    dm = np.array(df.mean(axis=1))    # Mean of the d_f
    ym=ym.reshape(ym.shape[0],1)    
    dm=dm.reshape(dm.shape[0],1)    
    dmf = yf - ym
    ddf = df - dm
    
    Cmd_f = (np.dot(dmf,ddf.T))/(num_ens-1);  # The cros-covariance matrix
    Cdd_f = (np.dot(ddf,ddf.T))/(num_ens-1);  # The auto covariance of predicted data
    
    # Perturb the vector of observations
    R = linalg.cholesky(CD,lower=True) #Matriz triangular inferior
    U = R.T   #Matriz R transposta
    p , w =np.linalg.eig(CD)
    
    aux = np.repeat(Z,num_ens,axis=1)
    mean = 0*(Z.T)

    noise=np.random.multivariate_normal(mean[0], np.eye(len(Z)), num_ens).T
    d_obs = aux+math.sqrt(alpha)*np.dot(U,noise)  
    
    # Analysis step
    u, s, vh = linalg.svd(Cdd_f+alpha*CD); v = vh.T
    diagonal = s
    for i in range(len(diagonal)):
        if (sum(diagonal[0:i+1]))/(sum(diagonal)) > varn:
            diagonal = diagonal[0:i+1]
            break
    
    u=u[:,0:i+1]
    v=v[:,0:i+1]
    ss = np.diag(diagonal**(-1))
    K=np.dot(Cmd_f,(np.dot(np.dot(v,ss),(u.T))))
    # Use Kalman covariance
    if len(corr)>0:
        K = corr*K
        
    ya = yf + (np.dot(K,(d_obs-df)))
    m_ens = ya
    return m_ens


# Create the headfile and budget file objects
#headobj = bf.HeadFile(modelname+'.hds')
#times = headobj.get_times()
#cbb = bf.CellBudgetFile(modelname+'.cbc')

# Plot the head versus time
#idx = (0, int(nrow/2) - 1, int(ncol/2) - 1)
#idx = (0, 20, 20)
#ts = headobj.get_ts(idx)
plt.subplot(1, 1, 1)
ttl = 'Head at cell ({0},{1},{2})'.format(idx[0] + 1, idx[1] + 1, idx[2] + 1)
plt.title(ttl)
plt.xlabel('time')
plt.ylabel('head')
for i in range(1,nr+1):  
    plt.plot(tss[:, 0], tss[:, i], '-')
plt.savefig('tutorial2-ts.png')