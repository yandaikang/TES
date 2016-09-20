import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.optimize import fsolve
from scipy.optimize import brenth
import warnings
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

warnings.filterwarnings("error")

'''
Tc = 0.0941
R_nor = 9e-3 #ohm, TES normal resistance, based on the IV curve fitting result06
I0 = 55e-3
'''

Tc = 0.1063 #K, transition temperature
R_nor = 8.8e-3 #ohm, TES normal resistance, based on the IV curve fitting result06
I0 = 20e-3
Gc = 124e-12 #W/K
n = 2.45

CI = 0.5
CR = 1

#T_list = [0.095]
T_list = np.arange(88,110,.05)*1e-3
I_list = np.arange(0, 400e-6, 5e-6)
T_list, I_list = np.meshgrid(T_list, I_list)

Is2 = (CI * I0)**2 * (1 - T_list/Tc)**3
for i in range(len(Is2)):
	for j in range(len(Is2[i])):
		if Is2[i,j] < 0:
			Is2[i,j] = 0

Is = Is2**0.5


V = (I_list - Is) * CR*R_nor
for i in range(len(V)):
	for j in range(len(V[i])):
		if V[i, j] < 0:
			V[i,j] = 0



R_array = np.divide(V, I_list, out=np.zeros_like(V), where=I_list!=0)
P_array = I_list**2 * R_array

def FindAPoint(R_array):
	l = R_array - 0.8*R_nor #locate at 80% Rn
	lmin = np.min(np.min(abs(l), axis=1), axis=0)


	Indi, Indj = np.where(l == lmin)

	return Indi[0], Indj[0]

Indi, Indj = FindAPoint(R_array)
#print Indi, Indj

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(T_list*1e3, I_list*1e6, R_array/R_nor, cmap=cm.winter_r, linewidth=0.2, alpha = .5)
#ax.plot_surface(T_list*1e3, V*1e6, P_array*1e12, cmap=cm.winter_r, linewidth=0.2)
#ax.plot_surface(T_list*1e3, I_list*1e6, P_array*1e12, cmap=cm.winter_r, linewidth=0.2)

ax.set_xlabel('T [mK]')
ax.set_ylabel('I [uA]')
#ax.set_ylabel('V [uV]')
ax.set_zlabel('R0/Rn')
#ax.set_zlabel('P [pW]')

#plt.plot(T_list, PT_P)

#plt.subplots_adjust(left=0.1, right=0.7, top=0.9, bottom=0.1)
#plt.legend(bbox_to_anchor=(1.5, 1.))
#plt.suptitle('Simulate\n\n'+r'Gc=%.2fpW/K, Tc=%.2fmK, n=%.2f'%(Gc*1e12, Tc*1e3, n))



#========================================================================
def FuncCompare1(T, V, Tb, Tc, R_nor, n, I0, CI, CR, Gc):
	y = V * (CI * I0*((1-T/Tc)**1.5) + V/(CR*R_nor)) - Gc/n/(Tc**(n-1)) *(T**n - Tb**n)
	return y

def FuncCompare2(T, V, Tb, Tc, R_nor, n, I0, CI, CR, Gc):
	y = V**2 / (CR*R_nor) - Gc/n/(Tc**(n-1)) *(T**n - Tb**n)
	return y


def CalCompare(T_root, V_list):
	T_list = np.empty(len(V_list))
	R_list = np.empty(len(V_list))
	I_list = np.empty(len(V_list))
	Is_list = np.empty(len(V_list))

	for i in range(len(V_list)):
		V =  V_list[i]
		args = (V, Tb, Tc, R_nor, n, I0, CI, CR, Gc)
		x0 = T_root

		try:
			T_root = fsolve(FuncCompare1, x0, args)[0]
			R = V/(CI * I0*((1 - T_root/Tc)**1.5) + V/(CR*R_nor))
			Is = CI*I0*((1 - T_root/Tc)**1.5)
			I = CI * I0*((1 - T_root/Tc)**1.5) + V/(CR*R_nor)

	
		except RuntimeWarning:
			T_root = fsolve(FuncCompare2, x0, args)[0]
			R = CR*R_nor
			Is = 0
			I = V/R


		T_list[i] = T_root
		R_list[i] = R
		I_list[i] = I
		Is_list[i] = Is

	return T_list, R_list, I_list, Is_list

Vmax = .145 #uV
Tb = 88e-3
T_root = Tb
V_list = np.arange(1.5e-3, Vmax, 0.001) * 1e-6
[T_list, R_list, I_list, Is_list] = CalCompare(T_root, V_list)
l = 150
ax.plot(T_list[:l]*1e3, Is_list[:l]*1e6, 0, color='b', linewidth=1)
ax.plot(T_list[:l]*1e3, I_list[:l]*1e6, R_list[:l]/R_nor, color='r', linewidth=1)

plt.show()
#sys.exit()

'''
f = open('PT_80Rn_bennett.pt', 'w')
for i in range(len(T_list)):
	value = str(T_list[i])+'\t'+str(PT_P[i])+'\n'
	f.write(value)
f.close()
'''