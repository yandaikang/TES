#This code runs the TES power v.s. bath temperature fitting, to find Tc, Gc and n.

import numpy as np
import math
import h5py
from scipy import signal
import struct;
import sys, os
import csv
import glob
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('figure', facecolor = 'w')

#==============================================================
def funPT(Tb, Gc, Tc, n):
	#the function used for fitting
	y = Gc/(n*Tc**(n-1))*(Tc**n - Tb**n)
	return y

def fitPT(funPT, x, y, p0):
	#fitting
	params = opt.curve_fit(funPT, x, y, p0)
	par = params[0] #fitted parameters
	cov = params[1]	#covariance of the parameters
	cov_Gc = np.sqrt(cov[0][0])
	cov_Tc = np.sqrt(cov[1][1])
	cov_n = np.sqrt(cov[2][2])

	#print par
	#print cov_Gc, cov_Tc, cov_n
	#print cov_Gc/par[0], cov_Tc/par[1], cov_n/par[2]

	return par, [cov_Gc, cov_Tc, cov_n]


Resname='Res8'
fname='07-19-2016/PT_'+Resname+'_80Rn.pt' #name of the PT file
f = open(fname, 'r')

Temp=[]
Power=[] #initiate the list for reading files
TempOmit=[]
PowerOmit=[]
for line in f:
	value = line.split('\t')
	if float(value[0]) in [98,99,100,101,102,103, 107,108,109,110]:
		#the points that you don't want to include in the fitting is saved in the --Omit list
		TempOmit.append(float(value[0]))
		PowerOmit.append(float(value[1]))
	else:
		Temp.append(float(value[0]))
		Power.append(float(value[1]))
f.close()

Temp=np.array(Temp)*1e-3 #mK convert to K
Power=np.array(Power)
TempOmit=np.array(TempOmit)*1e-3 #mK convert to K
PowerOmit=np.array(PowerOmit)

#data fitting
p0 = np.array([160e-12, 0.102, 1])
params, cov = fitPT(funPT, Temp, Power, p0)
Gc = params[0]
Tc = params[1]
n = params[2]

#plot the fit line
Temp_fit = np.arange(0.08, 0.11, 0.001)
Power_fit = funPT(Temp_fit, Gc, Tc, n)



plt.scatter(Temp*1e3, Power*1e12, color = 'b')
plt.scatter(TempOmit*1e3, PowerOmit*1e12, color = 'y') #the data points that are not included in fitting are plotted yellow
plt.plot(Temp_fit*1e3, Power_fit*1e12) #also plot the fitting line

plt.xlabel('Temperature [mK]')
plt.ylabel('$P_{tes}$ [pW]')
plt.axhline(y=0, color='k') #draw a horizontal line at y = 0, to see when P = 0

plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.suptitle(Resname+'\n\n'+r'G=%.2f $\pm$ %.2f pW/K, Tc=%.2f $\pm$ %.2f mK, n=%.2f $\pm$ %.2f'%(Gc*1e12, cov[0]*1e12, Tc*1e3, cov[1]*1e3, n, cov[2]))
plt.show()

