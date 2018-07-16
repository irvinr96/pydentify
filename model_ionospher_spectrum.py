#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 17:32:54 2018

@author: Robert Irvin
"""
import numpy as np
from dregion_spectrum import DregionSpectra
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.constants
import scipy.fftpack as fft


amu = scipy.constants.atomic_mass

wavenumber = 2*np.pi*(449.3e6 / scipy.constants.c)

M_H = 1.00794*amu
M_He= 4.00260*amu
M_O = 15.9994*amu
M_N = 14.0067*amu
M_Ar= 39.948*amu
M_O2= 2*M_O
M_N2= 2*M_N


b1 = '/Users/E31345/Desktop'
b2 = '/Volumes/scratch/AMISR_Work/spectra_Dregion'
basepath = b1
files =glob.glob(os.path.join(basepath,'20170508.001/datfiles/*.h5'))
dreg = DregionSpectra(files) 


# read in frequency and up beam altitude  (km) from this 
# mswinds 23 experiment 
altitude = dreg.get_altitude(dreg.times[0,0])[0]/1000
frequency = dreg.get_frequency(dreg.times[0,0])

infile = open('empirical_data/msise-90.txt','r')

h= infile.readlines()
infile.close()

idx = h.index('Selected output parameters:\n')

h = h[idx+1:]

h = [s.strip() for s in h] 
varss = h[0:10]

forms = []
for i in range(10):
    forms.append('f4')

header = tuple([i.split()[1].strip(',') for i in varss])
arr  = np.loadtxt('empirical_data/msise-90.txt',skiprows = 37,dtype =
                  {'names': header ,'formats': tuple(forms)})


plt.plot(np.log10(arr['Mass_density']),arr['Height'])
plt.xlabel('Mass Density log10(g / cm^-3)')
plt.show()
plt.plot((arr['Temperature_neutral']),arr['Height'])
plt.xlabel('Temperature Neutral (K)')
plt.show()
# we want to get the total neutral number density vs. height
# this is just the sum over all the species in the model

neutral_number_density = np.zeros(len(arr))
for i in range(len(arr)):
    neutral_number_density[i] = arr[i]['O']+arr[i]['N2']+arr[i]['O2']+arr[i]['He']+arr[i]['H']+arr[i]['N']+arr[i]['Ar']
plt.plot(np.log10(neutral_number_density), arr['Height'])   
plt.xlabel('Neutral Number Density log10( cm^-3)')
plt.show()
# Now let's calculate the scale height using definition for multiple species
# H= kT / <m>g
scale_height = np.zeros(len(arr))
mean_m =np.zeros(len(arr))

for i in range(len(arr)):
    dum = arr[i]
    mean_m[i] = (M_O*arr[i]['O']+M_N2*arr[i]['N2']+M_O2*arr[i]['O2']+M_He*arr[i]['He']+M_H*arr[i]['H']+M_N*arr[i]['N']+M_Ar*arr[i]['Ar'])/neutral_number_density[i]

    scale_height[i] = .001*(scipy.constants.k * arr[i]['Temperature_neutral'])/(9.8*mean_m[i])


plt.plot((mean_m), arr['Height'])   
plt.xlabel('Mean species mass (kg))')
plt.show()

plt.plot(scale_height, arr['Height'])   
plt.xlabel('Multi species scale height (km)')
plt.show()


# define the ion neutral collision frequency. 
# reference (Beharrell and Honary 2008)
incf = np.logspace(2,0,36)
incf = np.pad(incf,(65,49),mode = 'constant')
I = np.where(incf == 0.0) ; incf[I] = np.nan; incf = np.ma.masked_where(np.isnan(incf),incf)
def IDC(z):
    return 1/ incf[z]

def gamma(z):
    return 2*wavenumber**2*IDC(z)

# define the equation for the power spectral density
def PSD(f,z):
    return (gamma(z))/(2*np.pi*np.abs(f)+(gamma(z))**2)

# altitude to generate spectra
Spectra = np.zeros((len(arr),frequency.shape[0]))
gam = np.zeros(len(arr))
for i in range(len(arr)):
    aspec = PSD(frequency,i)
    Spectra[i] = aspec/np.max(aspec)
    gam[i] = gamma(i)



for alt in [65,75,85,95]:

    Spectra = Spectra / np.max(Spectra)
    plt.plot(frequency, Spectra[alt],label=str(alt))
width =  gamma(alt)
plt.legend()
plt.show()
# calculate moments
zm = np.sum(Spectra, axis =  1)
zm[np.where(zm==0.0)]=np.nan ; zm = np.ma.masked_where(np.isnan(zm),zm)
fm = 0
sm = np.sqrt(np.sum(frequency**2 * Spectra,axis=1)/zm)
sm  = np.ma.masked_where(np.isnan(sm),sm)
plt.plot(sm,arr['Height'], label = '2nd moment')
plt.plot(2*gam, arr['Height'],label = 'FWHM (2*gam)')
plt.legend()
plt.show()

def gam_to_sw(gamma, f):
    return 2*()

"""
for i in range(10):
    locals()[varss[i].split()[1].strip(',')] = []
"""


gamma = 100
times = np.linspace(-.256,.256 , 255)
acf = np.exp(-gamma*np.abs(times))
plt.plot(times,acf); plt.show()
spectrum = fft.fft(acf)
frequency = fft.fftfreq(255)
plt.plot(frequency,np.abs(spectrum),'ro'); plt.ylim(0);plt.show()

# doppler shift
gamma = 500
acf = np.exp(-gamma*np.abs(times) +1j*300*times)
spectrum = fft.fft(acf)
plt.plot(frequency,np.abs(spectrum),'ro');plt.ylim(0);plt.show()