#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 13:46:50 2018

@author: Robert Irvin
"""
""" Generate Gamma - Second moment table"""
import numpy as np

import scipy.constants
import scipy.fftpack as fft
from scipy.interpolate import interp2d
from os.path import join,exists
import os
import datetime as dt
import matplotlib
current_cmap = matplotlib.cm.inferno
current_cmap.set_bad('white',1.)
import matplotlib.pyplot as plt
basepath = '/Users/E31345/pydentify' #where to save plots
def table(gam, dop_shift):
    """Calculates first and second moments from an aliased 
    lorentzian spectrum generated for a given doppler shift and gamma"""
    
    gamma = gam
    doppler_shift = dop_shift #in angular frequency
    times = np.linspace(-.256,.256 , 255) # times matching the PFISR experiment
    
    if type(gam)==np.ma.core.MaskedConstant or type(dop_shift)==np.ma.core.MaskedConstant:
        return np.nan,np.nan
    
    
    def tri(N):
        """generates triangular window array"""
        arr = np.zeros(N+1)
        for i in range(N+1):
            arr[i]  =  ((N-i))/float(N)
        return np.pad(arr,(len(arr)-1,0),'reflect')
    
       # len(times) = 2*N +1
    tri_win = tri((len(times)-1)/2)
    # generate acf. needs to be multiplied by a triangular window
    acf = tri_win * np.exp(-gamma*np.abs(times) + 1j*2*np.pi*doppler_shift*times)

    spectrum = fft.fft(acf)
    frequency = fft.fftfreq(times.shape[0],d=.002) # 2ms IPP
    
    #Calculate the spectra moments
    zm = np.sum(np.abs(spectrum))
    fm = np.sum(np.abs(spectrum)*frequency)/zm
    sm = np.sqrt(np.sum((frequency-fm)**2 * np.abs(spectrum))/zm)
    return  sm,fm

def new_table(gam,dop_shift):

    #gam, dop_shift = np.array((80)),np.array((25))
    """Like table expcet for general arrays of doppler shift and gamma"""
    gamma, doppler_shift = np.meshgrid(gam,dop_shift)  
    times = np.linspace(-.256,.256 , 255) # times matching the PFISR experiment
    
    
    
    def tri(N):
        """generates triangular window array"""
        arr = np.zeros(N+1)
        for i in range(N+1):
            arr[i]  =  ((N-i))/float(N)
        return np.pad(arr,(len(arr)-1,0),'reflect')
    
       # len(times) = 2*N +1
    tri_win = tri((len(times)-1)/2)
    
    #expontential factor
    exponent = -np.einsum('...i,j->...ij',gamma,np.abs(times)) + 1j*2*np.pi*np.einsum('...i,j->...ij',doppler_shift,times)
    
    # generate acf. needs to be multiplied by a triangular window
    acf = tri_win * np.exp(exponent)
    
    spectrum = fft.fft(acf)
    frequency = fft.fftfreq(times.shape[0],d=.002) # 2ms IPP
    
    #Calculate the spectra moments
    zm = np.sum(np.abs(spectrum),axis=-1)
    fm = np.sum(np.abs(spectrum)*frequency,axis=-1)/zm
    
    f_bar =np.repeat(fm[:,:,np.newaxis],255,axis=-1) # for broadcsting
    
    sm = np.sqrt(np.sum((frequency-f_bar)**2 * np.abs(spectrum),axis=-1)/zm)
    return sm,fm

def plot_some_curves():
    # these first two functions are some attempted analytic relations
    def spec_width(gamma, f_0):
        return np.sqrt((gamma*f_0/np.pi**2)  - 0.5*(gamma**2/np.pi**3)*np.arctan(2*np.pi*f_0/gamma))

    def f_bar(gamma, f_0, f_max):
        term1  = gamma*np.log( (4*np.pi**2*(f_max-f_0)**2 +gamma**2)   /  (4*np.pi**2*(f_max + 
               f_0)**2 +gamma**2))/(4*np.pi**3)
        term2 = (f_0/np.pi)*np.arctan(2*np.pi*(f_max-f_0)/gamma)
        term3 = - (f_0/np.pi) * np.arctan(-2*np.pi*(f_max-f_0)/gamma)
        return  term1+term2+term3
    # generate figure, doppler shifts
    fig,ax = plt.subplots(3,1,figsize = (10,8))
    dopp = np.linspace(0,180,num=5)
    colors = ['b','g','r','c','y','k',]
    #plot moments vs. doppler shift for different doppler shifts, denoted
    # by the colored curves in the plots
    for i,dop in enumerate(dopp):
        gam_array = np.linspace(1,500 , 1000)
        fm_array = np.array([table(j, dop)[1] for j in gam_array])
        fm_ther  = f_bar(gam_array , dop, f_max = 250)
        sm_array = np.array([table(j, dop)[0] for j in gam_array])
        if dop == 0:
            sm_0 = sm_array

        ax[0].plot(2*gam_array,sm_array,'{}{}{}'.format(colors[i],'',''), label = str(dop)+'Hz')
        ax[1].plot(2*gam_array,fm_array,'{}{}{}'.format(colors[i],'',''), label = str(dop)+'Hz')
        ax[1].plot(2*gam_array,fm_ther,'{}{}{}'.format(colors[i],'','--'), label = str(dop)+'Hz')
    
    # compare attempted analytic relation to actual for doppler of 0  
    sw  = np.array([spec_width(j,250) for j in gam_array])
    ax[2].plot(2*gam_array,sw , label = 'Analytic Relation')
    ax[2].plot(2*gam_array,sm_0 , label = 'Actual') ; ax[2].legend(loc='lower right')
   
    ax[0].legend(loc = 'lower right')
    ax[2].set_xlabel('2 * Gamma (FWHM)')
    
    ax[0].set_ylabel('Spec Width from 2nd moment (Hz)')
    ax[1].set_ylabel('First Moment (Hz)')
    ax[2].set_ylabel('2nd moment w/ 0 Dop. (Hz)')
    
    string = 'oneD_moments_vs_gamma.png'
    path = join(basepath,string)
    
    plt.savefig(path,dpi=300)
    
    plt.show()
    plt.close(fig)


def forward_interp2d(resolution, kind):
    # Here we play with interp2d, generate doppler and gamma arrays
    doppler_array = np.linspace(0,200,int(200/resolution)+1)
    gamma_array   = np.linspace(1,500,500) 
   
    
    # generate d2,g2 1D arrays that make ordered pairs of gamma and doppler    
    d2 = np.repeat(doppler_array,gamma_array.shape[0]) 
    g2 = np.tile(gamma_array, doppler_array.shape[0])
    #iniitlaize moment arrays
    x_bar , del_xsquare = np.zeros((g2.shape)),np.zeros((g2.shape))
    
    for i,dop in enumerate(d2):
        sm, fm = table(gam = g2[i] , dop_shift=dop)
        
        x_bar[i] = fm ;del_xsquare[i] = sm
    
    f = interp2d(g2 , d2, x_bar,kind=kind)   # sanity check (gamma,doppler) --> (fm,sm)
    
    g = interp2d(g2 , d2, del_xsquare,kind=kind)   # sanity check (gamma,doppler) --> (fm,sm)
    

    #generate pcolormesh figures
    x,y = np.linspace(0,500,5001),np.linspace(0,200,2001) #fine resolution doppler and gamma

    
    first_mom = f(x,y)          # generate moments for each doppler/gamma 
    second_mom= g(x,y)          # ordered pair
    
    fig,ax = plt.subplots(1,2,figsize=(10,8)) #initialize figure
    # first block is first moment figure routine. pcolormesh and colorbar
    im0 = ax[0].pcolormesh(x,y,first_mom,cmap='inferno',vmin=0,vmax=200)
    ax[0].contour(x,y,first_mom,cmap='viridis_r',vmin=0,vmax=200)
    ax[0].set_ylabel('Doppler shift (Hz)');ax[0].set_xlabel('Gamma/HWHM (Hz)')
    ax[0].set_title(kind+' First Moment (Hz) resolution ='+str(resolution)+'Hz')
    fig.colorbar(im0,ax=ax[0])

    # second block is second moment figure routine
    im1 = ax[1].pcolormesh(x,y,(second_mom),cmap='inferno',vmin=0,vmax=170)
    ax[1].contour(x,y,(second_mom),cmap='viridis_r',vmin=0,vmax=170)
    ax[1].set_ylabel('Doppler shift (Hz)');ax[1].set_xlabel('Gamma/HWHM (Hz)')
    ax[1].set_title(kind+' Second Moment (Hz) resolution ='+str(resolution)+'Hz')
    fig.colorbar(im1,ax=ax[1])
   
    if not exists(join(basepath,'forward_interp2d')):
        os.mkdir(join(basepath,'forward_interp2d'))
    #   kind+'_'+parameter+'_inv_interp_res_'+str(resolution)+'.png'
    string = '%s_for_interp2d_res_%s.png'%(kind,str(resolution))
    plt.savefig(join(basepath,'forward_interp2d',string))    
    
    plt.show(fig)
    plt.close(fig)
    
# These figures seem to agree quite well with the prior figures. 
# so interpolation in the forward direction seems to be working, let 
# us try the inverse problem now

def inverse_interp2d(parameter , resolution, kind ):
    """ parameter is a string, either 'doppler' or 'gamma' depending on what you want to plot
    Resolution is an integer that denotes the number of doppler
     shift frequencies used in calculation. Note doppler shift always ranges from
     0 to 200 Hz"""

    # Here we generate doppler and gamma arrays
    doppler_array = np.linspace(0,200,int(200/resolution)+1)
    gamma_array   = np.linspace(1,500,500)
    
    # 1d arrays denoting ordered pairs
    d2 = np.repeat(doppler_array,gamma_array.shape[0]) 
    g2 = np.tile(gamma_array, doppler_array.shape[0])
    
    x_bar , del_xsquare = np.zeros((g2.shape)),np.zeros((g2.shape)) #initialize first/second moment arays
    
    for i,dop in enumerate(d2):   #loop through doppler array/gamma, calculating moments at each
        sm, fm = table(gam = g2[i] , dop_shift=dop)
        
        x_bar[i] = fm ;del_xsquare[i] = sm
 
    
    
    
    #first and second moment arrays for pcolormesh plotting
    x2 = np.linspace(0,200,201)    
    del2 = np.linspace(0,170,171)
    # generate scatter plot showing density of points 
    plt.scatter(x_bar,del_xsquare)
    plt.xlim(x2[0],x2[-1])      ;plt.xlabel('Second moment (Hz)')
    plt.ylim(del2[0],del2[-1]) ; plt.ylabel('First moment (Hz)')
    plt.title('Point density in moment phase space')
    plt.show()
    plt.close()
    
    
    
    # generate f, the figure title, and the colorbar limits based on parameter
    
    if parameter == 'doppler':
        title = kind+'Doppler shift (Hz) resolution ='+str(resolution)
        #use interp2d
        f = interp2d(x_bar ,del_xsquare , d2,kind= kind,fill_value= np.nan) #(fm,sm) -> dop shift
        cmax = doppler_array[-1]
    elif parameter == 'gamma':
        title = kind+' Gamma/HWHM (Hz) resolution ='+str(resolution)
        f = interp2d(x_bar ,del_xsquare , g2, kind=kind,fill_value=np.nan) #(fm,sm) -> gamma
        cmax = gamma_array[-1]
        
    x,y = np.meshgrid(x2,del2) #generate meshgrid 
    z_out = f(x2,del2)              # calculate parameter using f
    

    
    z = np.ma.masked_where(np.isnan(z_out),z_out)
    
    fig,ax = plt.subplots(figsize=(10,8))
    im = ax.pcolormesh(x,y,z,cmap=current_cmap,vmin=0,vmax = cmax)
    ax.set_xlabel('First Moment (Hz)');ax.set_ylabel('Second Moment (Hz)')
    fig.suptitle(title)
    fig.colorbar(im)
    
    if not exists(join(basepath,'inv_interp2d',kind,str(resolution))):
        os.mkdir(join(basepath,'inv_interp2d',kind,str(resolution)))
    
    string ='%s_%s_inv_interp2d_res_%s.png'%(kind,parameter,str(resolution))
    plt.savefig(join(basepath,'inv_interp2d',kind,str(resolution),string))
    plt.show(fig)
    plt.close(fig)
    del f
    return x2,del2,z_out


def error_plot(resolution,kind):

    fm,sm,doppler = inverse_interp2d('doppler',resolution,kind)
    fm,sm,gamma = inverse_interp2d('gamma',resolution,kind)
    #second,first = new_table(gamma.ravel(),doppler.ravel())
    
     #   f_gamma = interp2d(fm,sm, gamma)
      #  f_doppler = interp2d(fm,sm, doppler)
    #initialize difference array
    first_diff = np.zeros((sm.shape[0] , fm.shape[0]))
    second_diff= np.zeros((sm.shape[0] , fm.shape[0]))
    for i,first in enumerate(fm):
        for j,second in enumerate(sm):
            gam = gamma[j,i]
            dop = doppler[j,i]
            
            second_table,first_table = table(gam,dop)
            
            first_diff[j,i] = np.abs((first_table-first)/first)
            second_diff[j,i]= np.abs((second_table-second)/second)
      
    
    # mask array
    first_diff = np.ma.masked_where(np.isnan(first_diff),first_diff)
    second_diff = np.ma.masked_where(np.isnan(second_diff),second_diff)
    
    
    # now we have our second diff arrays lets plot them now
    fig,ax = plt.subplots(1,2,figsize=(10,8)) #initialize figure
    x,y = np.meshgrid(fm,sm)
    # first block is first moment figure routine. pcolormesh and colorbar
    im0 = ax[0].pcolormesh(x,y,np.log10(first_diff),cmap='inferno',vmin=-7,vmax=0)
    ax[0].set_ylabel('Second Moment (Hz)');ax[0].set_xlabel('First Moment (Hz)')
    ax[0].set_title(kind+' FM Error Fraction resolution ='+str(resolution)+'Hz')
    fig.colorbar(im0,ax=ax[0])
    
    # second block is second moment figure routine
    im1 = ax[1].pcolormesh(x,y,np.log10(second_diff),cmap='inferno',vmin=-7,vmax=0)
    ax[1].set_ylabel(' Second Moment (Hz)');ax[1].set_xlabel('First Moment (Hz)')
    ax[1].set_title(kind+' SM Error Fraction resolution ='+str(resolution)+'Hz')
    fig.colorbar(im1,ax=ax[1])
    
    string = '%s_table_interp2d_res_%s.png'%(kind,str(resolution))
    plt.savefig(join(basepath,'inv_interp2d',kind,str(resolution),string))
    
    plt.show(fig)
    plt.close(fig)

