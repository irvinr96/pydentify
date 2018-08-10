# pydentify
Tools for handling PFISR D-region mode spectra data

#### gamma_mom_sc.py
This program does two things. 

1: finds the spectral moment to Lorentzian parameter (doppler shift, gamma/HWHM) forward and inverse relations 

2: a first try at an automated EPP detection algorithm using these relations  

There are a number of functions in this script, this document explains how they work together, and any modifications that need to be made 

First, change the variable basepath to the desired output directory for the generated plots. This script also outputs many plots to the screen, comment these lines out if this is not desired or if running these functions in a loop. To get quick information and a simple example for a function type 'func?' into ipython console. Note: I have been using spyder 3.1.4 and python 2.7.13 

generate_table returns the inputs for griddata spectral moment to Lorentzian parameter relations. These are set to global variables in line 279 for use in the gd function. fitstat  
