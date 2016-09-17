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
class Res():
	"""define a class to store the data of a TES channel"""

	def __init__(self):
		self.Vbias = []
		self.phase = []	
		self.f = []

		self.Rbias = 9.96*1e3 #the bias resistor at room temperature
		self.Rsh = 300.*1e-6 #shunt resistor
		self.Iperiod = 8.9*1e-6 #TES current period, measured from Fr v.s. Isquid
		self.Rlead = 4.e-6 #lead resistance in series with the TES


	def ReadFile(self, filename, chan):
		self.Temp=float(filename.split('/')[1].split('_')[0].strip('mK')) 	#Temperature of the measurement, read from the datafile name
		self.incrs=filename.split('/')[1].split('_')[1] 	#this marks if the data is taken by voltage sweeping up or sweeping down, can be used as label in the plots
		self.Temp_incrs=str(self.Temp)+'mK_'+self.incrs 	#this marks if the data is taken by voltage sweeping up or sweeping down, can be used as label in the plots

		datafile = h5py.File(filename, 'r')
		self.Vbias = datafile['vsa'][chan]['item_0'].value[:-5]*1e-3 #number from the voltage source, convert mV into V
		self.phase = datafile['vsa'][chan]['item_1'].value[:-5] #demodulated phase
		self.I = self.phase*self.Iperiod/(2*np.pi) #conversion between phase and current, Iperiod is measured in advance
		self.f = datafile['vsa'][chan]['item_2'].value[0]*1e-9 #the resonant frequency for this channel
		datafile.close()


	def funLinear(self, x, a, b):
		y = a*x + b
		return y

	def fitLinear(self, funLinear, x, y, p0):
		#curve fitting function
		params = opt.curve_fit(funLinear, x, y, p0)
		return params[0]


	def FindIc(self):
		#find the Ic data point's index in the data list, by taking second order derivative
		#this function is not perfect, just works for the data taken so far
		#it sometimes has problem with the voltage-sweep-up data, because the Ic point is not sharp enough
		ddI = np.gradient(np.gradient(self.I))
		self.IcInd = list(ddI).index(ddI.min())

		#if self.IcInd>80: 	#for high temperature data, there's no Ic, but this gradient function will still give an Ic index
		#	self.IcInd=0 	#for this dataset, their "Ic index" is larger than 80, which is different from other transition data
							#for a different data set, the logic might be different

		

	def ExtendFit(self):
		#align the normal region--------------------------------------------------------
		#the transition & normal intersection is not distinct enought to be found out, so just use the last 250 points to fit the normal region 
		self.Vbias_nor = self.Vbias[-100:]
		self.I_nor = self.I[-100:]

		p0 = np.array([2e-3, 15.])	#initial guess for fitting purpose, the value is not crutial
		params = self.fitLinear(self.funLinear, self.Vbias_nor, self.I_nor, p0)
		self.slop_nor = params[0]
		self.delta_nor = params[1]

		self.Vbias_ext_nor = np.arange(-1., self.Vbias[-1], 0.02) 
		self.I_ext_nor = self.funLinear(self.Vbias_ext_nor, self.slop_nor, self.delta_nor)

		self.I -= self.delta_nor 	#shift all the data
		self.I_ext_nor -= self.delta_nor 	#shift all the data, to make the normal branch extends through (0,0)
		#-------------------------------------------------------------------------------


		#align the superconducting region-----------------------------------------------
		if self.IcInd==0:
			#when at high temperature, the TES stay normal all the way
			self.delta_sup = 0

			self.Vbias_ext_sup = 0
			self.I_ext_sup = 0

		else:
			self.Vbias_sup = self.Vbias[:self.IcInd]
			self.I_sup = self.I[:self.IcInd]

			p0 = np.array([1e-3, -15.])
			params = self.fitLinear(self.funLinear, self.Vbias_sup, self.I_sup, p0)
			self.slop_sup = params[0]
			self.delta_sup = params[1]

			self.Vbias_ext_sup = np.arange(-1, self.Vbias[self.IcInd], 0.2)
			self.I_ext_sup = self.funLinear(self.Vbias_ext_sup, self.slop_sup, self.delta_sup)	#the fitted line
			#the superconducting branch needs to be shifted also, 
			#but it is done in a separate function (Shift), in order to look at the non-shifted data
		#--------------------------------------------------------------------------------


	def Shift(self):
		#the function that shifts the superconducting branch
		self.I[:self.IcInd] -= self.delta_sup
		self.I_ext_sup -= self.delta_sup


	def Scale(self):
		#The being measured data are Vbias and Ites, needed are Vtes and Ites
		#Vtes = (Vbias/Rbias - Ites)*Rsh - Ites * Rlead
		#the scaled data superconducting branch should be a vertical line
		self.V = (self.Vbias/self.Rbias - self.I)*self.Rsh - self.I*self.Rlead
		self.V_ext_nor = (self.Vbias_ext_nor/self.Rbias - self.I_ext_nor)*self.Rsh - self.I_ext_nor*self.Rlead

		self.R_nor = np.mean(np.gradient(self.V_ext_nor)/np.gradient(self.I_ext_nor)) 	#from the scaled data, TES normal resistance can be calculated by R = V/I

		self.Ic = self.I[self.IcInd] 	#the critical current value

		self.R = self.V/self.I 	#TES resistance at each bias point


	def FindAPoint(self):
		#this function is to find one point, for later PT plotting.
		#at first I picked the point at a fixed TES voltage as the following two lines:
		#l = self.V - 40e-9 #locate at 40 uV
		#self.VInd = list(abs(l)).index(min(abs(l)))

		#then based on NIST's suggestion, I changed to pick the point at 80% TES normal resistance
		l = abs(self.R - 0.8*self.R_nor) #locate at 80% Rn
		self.VInd = list(l).index(min(l))


	def Power(self):
		#the Joule power into the TES P=I*V
		self.P = self.I * self.V


colormap=['r', 'y', 'g', 'b', 'c', 'm', 'k'] 	#gives a list of color, used for plotting
def Exct(chan, Resname, filename, color):
	#this is the main function that get and plot all the data values from the Res class
	incrs=filename.split('/')[1].split('_')[1]
	Res24 = Res()

	Res24.ReadFile(filename, chan) 	#read file
	Res24.FindIc() 	#find the Ic index, which separates the superconducting branch with transition part


	Res24.ExtendFit() 	#fit the normal and superconducting branch respectively

	#you can plot and check some data like the following commands
	#plt.plot(Res24.Vbias, Res24.I*1e6, linestyle='-', color = color, label = str(Res24.Temp)+' mK')
	#plt.plot(Res24.Vbias*1e6, Res24.I*1e6, linestyle='--', color = color)
	#plt.plot(Res24.Vbias_ext_sup*1e6, Res24.I_ext_sup*1e6, linestyle=':')
	#plt.plot(Res24.Vbias_ext_nor*1e6, Res24.I_ext_nor*1e6, linestyle=':', color = color)

	
	Res24.Shift() 	#shift the superconducting branch
	Res24.Scale()	#scale Vbias to Vtes, and do some related calculation
	Res24.Power()	#calculate the Joule power P = I*V
	Res24.FindAPoint() #find the point for later PT plot

	#below are something I would look for an IV measurement
	plt.plot(Res24.Vbias, Res24.I*1e6, linestyle='-', marker = '.', color = color, label = str(Res24.Temp)+' mK')
	#plt.plot(Res24.Vbias/Res24.Rbias*1e3, Res24.I*1e6, linestyle='--', color = color, label = str(Res24.Temp)+' mK')
	
	#plt.plot(Res24.Vbias_ext_sup, Res24.I_ext_sup*1e6, linestyle=':', color=color)
	#plt.plot(Res24.Vbias_ext_nor, Res24.I_ext_nor*1e6, linestyle=':', color = color)

	#plt.plot(Res24.V*1e9, Res24.I*1e6, linestyle='-', marker='.', color = color)#, label = Resname )#= str(Res24.Temp)+' mK')
	#plt.plot(Res24.V_ext_nor*1e9, Res24.I_ext_nor*1e6, linestyle='--', marker='', color = color)#, label = Res24.R_nor)

	#plt.plot(Res24.V*1e9, Res24.P*1e12, linestyle='-', marker='.', color = color)
	#plt.plot(Res24.V[Res24.VInd]*1e9, Res24.P[Res24.VInd]*1e12, linestyle=' ', marker='o', color = color)

	#plt.plot(Res24.V*1e9, Res24.R*1e3, linestyle='-', marker='.', color = color, label = str(Res24.Temp)+' mK')
	#plt.plot(Res24.V[Res24.VInd]*1e9, Res24.R[Res24.VInd]*1e3, linestyle='', marker='o', color = color)#, label = Resname )#= str(Res24.Temp)+' mK')

	#plt.xlabel('I$_{bais}$ [mA]', size=15)
	#plt.xlabel('V$_{tes}$ [nV]', size=15)
	plt.xlabel('V$_{bias}$ [V]', size=15)

	plt.ylabel('I$_{tes}$ [$\mu$A]', size=15)
	#plt.ylabel('P$_{tes}$ [pW]', size=15)
	#plt.ylabel('R$_{tes}$ [m$\Omega$]', size=15)
	return Res24


filelist = glob.glob('07-19-2016/*.h5') 	#a list of files that you may want to plot in a same figure
Num = len(filelist)	#number of files

chanlist=['keyint_192', 'keyint_193', 'keyint_194']  #name for Res-24, 10, 8, this is how they are saved in the h5 file
Reslist=['Res-24', 'Res-10', 'Res-8']
for j in range(2,3):
	#this loop choses the resonator channel (i.e. TES channel)
	chan=chanlist[j]	
	Resname = Reslist[j]
	Power=[]
	Temp =[]
	for i in range(Num):
		#this loop changes which file you want to plot
		filename=filelist[i]
		color=colormap[int(i%7)]
		Res24 = Exct(chan, Resname, filename, color)

		Power.append(Res24.P[Res24.VInd])
		Temp.append(Res24.Temp)

	Power=np.array(Power)
	Temp=np.array(Temp)
	#the Power and Temp can be used for PT fitting to get G and n parameters
	#if you want to do this fitting, un-comment the file saving command below to save Power and Temp in a file, then use PT_fit.py to run the fit


'''
f = open('07-19-2016/PT_Res8_80Rn.pt', 'w')
for i in range(len(Temp)):
	value = str(Temp[i])+'\t'+str(Power[i])+'\n'
	f.write(value)
f.close()
'''


#plt.plot(Temp, Power*1e12, linestyle='', marker='o')
#plt.xlabel('Temperature [mK]', size=15)
#plt.ylabel('P$_{tes}$ [pW]', size=15)

#plt.axvline(x=0, color='k')
#plt.axhline(y=0, color='k')

#plt.subplots_adjust(left=0.1, right=0.7, top=0.9, bottom=0.1)
#plt.legend(bbox_to_anchor=(1.5, 1.))
#plt.legend(loc='lower right')
plt.suptitle(Resname)#+', avg. R normal %.2f Ohm' % np.mean(Res24.R_nor))
plt.show()
