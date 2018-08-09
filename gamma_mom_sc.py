#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 13:46:50 2018

@author: Robert Irvin

Generate Gamma/Doppler - Moments table


Change basepath to where you want to save the plots"""

import numpy as np
import scipy.constants
import scipy.fftpack as fft
from scipy.interpolate import interp2d
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
from os.path import join,exists
import os
import datetime as dt
import matplotlib
current_cmap = matplotlib.cm.inferno #set masked values to white
current_cmap.set_bad('white',1.)
import matplotlib.pyplot as plt


basepath = '/Users/E31345/pydentify' #where to save plots for plot_some_curves, generate_table, and fitstat

def tri(N):
    """generates triangular window array of length 2N+1 used in table
    
    Example
    -------
    >>> tri(2)
    array([ 0. ,  0.5,  1. ,  0.5,  0. ])"""
    arr = np.zeros(N+1)
    for i in range(N+1):
        arr[i]  =  ((N-i))/float(N)
    return np.pad(arr,(len(arr)-1,0),'reflect')

def table(gam, dop_shift):
    """Returns second and first moments calculated from an aliased 
    lorentzian spectrum generated for a given gamma and doppler shift
    
    Example
    --------
    >>> table(80,25)
    (53.324606805301372, 23.053279708202361)"""
    
    gamma = gam
    doppler_shift = dop_shift #in angular frequency
    times = np.linspace(-.256,.256 , 255) # times matching the PFISR experiment
    
    if type(gam)==np.ma.core.MaskedConstant or type(dop_shift)==np.ma.core.MaskedConstant:
        return np.nan,np.nan

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


def plot_some_curves():
    """Plots detailing the relations between first/second moment 
    vs. gamma for various doppler shifts. Some analytical relations are
    tried (and are underestimated)
    
    >>> plot_some_curves()"""
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

def generate_table(resolution,kind):
    
    """Save plots of moments vs. gamma and doppler shift with contours for a given resolution in
    doppler shift and either 'cubic','nearest', or 'linear' interpolation
    
    Example
    --------
    >>> generate_table(1,'cubic')
    """
    # initialize gamma and doppler space
    doppler_array = np.linspace(0,200,int(200/resolution)+1)
    gamma_array   = np.linspace(1,50,500)**2 
    # generate d2,g2 1D arrays that make ordered pairs of gamma and doppler    
    d2 = np.repeat(doppler_array,gamma_array.shape[0]) 
    g2 = np.tile(gamma_array, doppler_array.shape[0])
    #iniitlaize moment arrays and calculate for each doppler, gamma ordered pair
    x_bar , del_xsquare = np.zeros((g2.shape)),np.zeros((g2.shape))
    
    for i,dop in enumerate(d2):
        sm, fm = table(gam = g2[i] , dop_shift=dop)
        
        x_bar[i] = fm ;del_xsquare[i] = sm
    
    # generate griddata for first and second moments
    x,y = np.meshgrid(np.linspace(0,500,501),np.linspace(0,200,201)) # gamma/doppler grid
    points = (d2,g2)
    grid_fm = griddata(points, x_bar,(y,x),method=kind)
    grid_sm = griddata(points, del_xsquare,(y,x),method=kind)
       
    ###### Forward interp (gam, dop)==>(sm,fm)##########
    fig,ax = plt.subplots(1,2,figsize=(10,8)) #initialize figure
    # first block is first moment figure routine. pcolormesh and colorbar
    im0 = ax[0].pcolormesh(x,y,grid_fm,cmap='inferno',vmin=0,vmax=200)
    c0 = ax[0].contour(x,y,grid_fm,cmap='viridis_r',vmin=0,vmax=200)
    labels = map(int,list(c0.levels))
    ax[0].clabel(c0,fmt='%1.1f')
    ax[0].set_ylabel('Doppler shift (Hz)');ax[0].set_xlabel('Gamma/HWHM (Hz)')
    ax[0].set_title(kind+' First Moment (Hz) resolution ='+str(resolution)+'Hz')
    fig.colorbar(im0,ax=ax[0])
    
    # second block is second moment figure routine
    im1 = ax[1].pcolormesh(x,y,(grid_sm),cmap='inferno',vmin=0,vmax=170)
    c1 = ax[1].contour(x,y,(grid_sm),cmap='viridis_r',vmin=0,vmax=170)
    ax[1].clabel(c1)
    ax[1].set_ylabel('Doppler shift (Hz)');ax[1].set_xlabel('Gamma/HWHM (Hz)')
    ax[1].set_title(kind+' Second Moment (Hz) resolution ='+str(resolution)+'Hz')
    fig.colorbar(im1,ax=ax[1])
       
    path = join(basepath,'griddata',kind,str(resolution))
    
    if not exists(path):
        os.makedirs(path)
    string = '%s_forward_griddata_res_%s.png'%(kind,str(resolution)+'Hz_res')
    plt.savefig(join(path,string),dpi=300)    
    plt.show(fig)
    plt.close(fig) 
     
    ######## Inverse interp (sm,fm)==>(gam,dop)########
    # first moment by second moment grid
    x,y = np.meshgrid(np.linspace(0,200,201),np.linspace(0,170,171))
    points = (del_xsquare,x_bar)
    grid_gam = griddata(points , g2 , (y,x),method=kind)
    grid_dop = griddata(points , d2 , (y,x),method=kind)
    
    # save these arrays for error plot
    grid_gam_out = grid_gam
    grid_dop_out = grid_dop
    
    # mask arrays
    grid_gam = np.ma.masked_where(np.isnan(grid_gam),grid_gam)
    grid_dop = np.ma.masked_where(np.isnan(grid_dop),grid_dop)

    fig,ax = plt.subplots(1,2,figsize=(10,8)) #initialize figure
    # first block is first moment figure routine. pcolormesh and colorbar
    im0 = ax[0].pcolormesh(x,y,grid_dop,cmap='inferno',vmin=0,vmax=200)
    c0 = ax[0].contour(x,y,grid_dop,cmap='viridis_r',vmin=0,vmax=200)
    ax[0].clabel(c0)
    ax[0].set_ylabel('Second moment (Hz)');ax[0].set_xlabel('First moment (Hz)')
    ax[0].set_title(kind+' Doppler shift (Hz) resolution ='+str(resolution)+'Hz')
    fig.colorbar(im0,ax=ax[0])
    
    # second block is second moment figure routine
    im1 = ax[1].pcolormesh(x,y,np.log10(grid_gam),cmap='inferno',vmin=-1,vmax=4)
    c1 = ax[1].contour(x,y,np.log10(grid_gam),cmap='viridis_r',vmin=-1,vmax=4)
    ax[1].clabel(c1)
    ax[1].set_ylabel('Second moment (Hz)');ax[1].set_xlabel('First moment (Hz)')
    ax[1].set_title(kind+' log10(Gamma/HWHM) (Hz) resolution ='+str(resolution)+'Hz')
    fig.colorbar(im1,ax=ax[1])
    
    string ='%s_inverse_griddata_res_%s.png'%(kind,str(resolution))
    plt.savefig(join(path,string),dpi=300)
    
    plt.show(fig)
    plt.close(fig) 
    
    ###### Scatter Plot Moment Phase Space #######
    
    plt.scatter(x_bar,del_xsquare)
    plt.xlim(x[0,0],x[0,-1])      ;plt.ylabel('Second moment (Hz)')
    plt.ylim(y[0,0],y[-1,0]) ; plt.xlabel('First moment (Hz)')
    plt.title('Point density in moment phase space %s res=%sHz'%(kind,str(resolution)))
    string ='%s_phasespace_griddata_res_%s.png'%(kind,str(resolution))
    plt.savefig(join(path,string),dpi=300)
    plt.show()
    plt.close()
    
    ######## Moment Error plot #########
    
    fm,sm,doppler = x[0,:],y[:,0],grid_dop_out
    fm,sm,gamma = x[0,:],y[:,0],grid_gam_out
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
    ax[0].set_title(kind+' log10 FM Error Fraction resolution ='+str(resolution)+'Hz')
    fig.colorbar(im0,ax=ax[0])
    
    # second block is second moment figure routine
    im1 = ax[1].pcolormesh(x,y,np.log10(second_diff),cmap='inferno',vmin=-7,vmax=0)
    ax[1].set_ylabel(' Second Moment (Hz)');ax[1].set_xlabel('First Moment (Hz)')
    ax[1].set_title(kind+' log10 SM Error Fraction resolution ='+str(resolution)+'Hz')
    fig.colorbar(im1,ax=ax[1])
    
    string = '%s_table_griddata_res_%s.png'%(kind,str(resolution))
    plt.savefig(join(path,string),dpi=300)
    
    plt.show(fig)
    plt.close(fig)

    return points, d2, g2
# run generate_table. points,d2 and g2 are global variables that are used in fitstat(exp)
points, d2, g2 = generate_table(1,'cubic')

def gd(sm,fm):
    """Inverse of table function. Returns arrays of gamma and doppler shifts given
    arrays of first and second moment
    
    Example
    -------
    >>> gd(np.array([50,100]),np.array([15,15]))
    (array([  70.09027157,  359.67831203]), array([ 16.0844093 ,  23.00540148]))"""
    gam = griddata(points, g2, (sm,np.abs(fm)))
    dop = griddata(points, d2, (sm,np.abs(fm)))
    return gam,dop*(np.sign(fm)) 

def datetime_bin(he):
    """return tuple of half hour datetime range 
    
    Example
    -------
    >>> import datetime as dt
    >>> x = dt.datetime.now()
    >>> x
    datetime.datetime(2018, 8, 8, 15, 33, 23, 420589)
    >>> datetime_bin(x)
    (datetime.datetime(2018, 8, 8, 15, 30), datetime.datetime(2018, 8, 8, 16, 0))"""
    
    td = dt.timedelta(minutes=30)
    floor = he.replace(minute=0,second=0,microsecond=0)
    if he - floor <td:
        return (floor,floor+td)
    elif  he - floor >= td:
        return (floor + td,floor+2*td) 

def fitstat(exp):
    """ 
    Example
    -------
    >>> fitstat('20170508.001')
    array([[ (datetime.datetime(2017, 5, 8, 3, 0), datetime.datetime(2017, 5, 8, 3, 30)),
        1],
       [ (datetime.datetime(2017, 5, 8, 3, 30), datetime.datetime(2017, 5, 8, 4, 0)),
        1],
       [ (datetime.datetime(2017, 5, 8, 4, 0), datetime.datetime(2017, 5, 8, 4, 30)),
        1],
       [ (datetime.datetime(2017, 5, 8, 4, 30), datetime.datetime(2017, 5, 8, 5, 0)),
        0],
       [ (datetime.datetime(2017, 5, 8, 5, 0), datetime.datetime(2017, 5, 8, 5, 30)),
        0],
       [ (datetime.datetime(2017, 5, 8, 5, 30), datetime.datetime(2017, 5, 8, 6, 0)),
        0]], dtype=object)"""
    
    #### read in moments/ altitude/ time arrays
    dname = exp
    spec_dregion_path = '/Volumes/scratch/AMISR_Work/spectra_Dregion' 
    momentdir = join(spec_dregion_path,dname,'moments')
    altitude_array,fm_array,noise_array,sm_array,utime_array,zm_array = np.load(join(momentdir,'altitude'))/1000,np.load(join(momentdir,
                'first_moment')),np.load(join(momentdir,'noise')),np.load(join(momentdir,
                'second_moment')),np.load(join(momentdir,'unixtime')),np.load(join(momentdir,'zero_moment'))


    
    # look at first beam
    fm_array = fm_array[:,0]
    sm_array = sm_array[:,0]
    
    # lay them out
    fms = fm_array.ravel()
    sms = sm_array.ravel()
    # calculate doppler shift and gamma for each ordered pair
    a,b = gd(sms,fms)  

    arr = np.array(map(table,a,b))
    print 'Done'
    
    
    #first column of arr is second moment , second colum
    # is first moment
    
    new_sm = arr[:,0].reshape(fm_array.shape)
    new_fm = arr[:,1].reshape(fm_array.shape)
    
    # mask arrays
    new_sm = np.ma.masked_where(np.isnan(new_sm),new_sm)
    new_fm = np.ma.masked_where(np.isnan(new_fm),new_fm)
    
    plt.pcolormesh(utime_array,altitude_array[0],new_sm.T,cmap='inferno',vmin=0,vmax=150)
    plt.colorbar(); plt.show(); plt.close()
    
    plt.pcolormesh(utime_array,altitude_array[0],new_fm.T,cmap='RdBu_r',vmin=-250,vmax=250)
    plt.colorbar(); plt.show(); plt.close()

    #plot gam and doppler
    
    new_gam = a.reshape(fm_array.shape)
    new_dop = b.reshape(fm_array.shape)
    
    # mask arrays
    new_gam = np.ma.masked_where(np.isnan(new_gam),new_gam)
    new_dop = np.ma.masked_where(np.isnan(new_dop),new_dop)
    
    plt.pcolormesh(utime_array,altitude_array[0],np.log10(new_gam).T,cmap='inferno',vmin=1,vmax=4)
    plt.colorbar(); plt.show(); plt.close()
    
    plt.pcolormesh(utime_array,altitude_array[0],new_dop.T,cmap='RdBu_r',vmin=-200,vmax=200)
    plt.colorbar(); plt.show(); plt.close()
    
    #calculate signal to noise ratio
    snr = 10*np.log10(zm_array/noise_array[:,:,np.newaxis])
    plt.pcolormesh(utime_array,altitude_array[0],snr[:,0].T,vmin=0,vmax=20,cmap='inferno')
    plt.colorbar(); plt.show(); plt.close()
    

    
    ###### Automated Algorithm Scale Height Fitting Code ##########
    # only look at up beam 
    ind_beam = 0
    
    alt_array = altitude_array[ind_beam]
    dB_cutoff = 15          # power threshold
    alt_cutoff = [60,120]  # relevant dregion altitudes
    alt_res = alt_array[1]-alt_array[0]
    
    # get indexes of the relavant altidues
    idxs = (np.abs(alt_array - alt_cutoff[0]).argmin(),
            np.abs(alt_array - alt_cutoff[1]).argmin())
    
    snr_beam =       snr[:,ind_beam, idxs[0]:idxs[1]]
    alt_array = alt_array[idxs[0]:idxs[1]]
    gam_out =       new_gam[:,idxs[0]:idxs[1]]
    
    # initialize array of fitting parameters
    popt_array = np.zeros((len(utime_array),4))
    
    for i,utime in enumerate(utime_array):
        #iterate through time axis, example start with first time
        time = i
        snr_loop = snr_beam[time]
        gam = gam_out[time]
    
    #    plt.plot(snr_loop,alt_array) ;plt.suptitle('snr');plt.show(); plt.close()
    #    plt.plot(alt_array,gam) ;plt.suptitle('gam');plt.show(); plt.close()
        
        
        I = np.where(snr_loop < dB_cutoff)
        gam[I] = np.nan
        alt_array[I]=np.nan
        
        
        
        def func(z,z_0,b,c,d):
            output = np.ones(z.shape)*c*np.exp(b*(d-z_0))
            
            inds = np.where( z < d)       
            output[inds] = c*np.exp(b*(z[inds]-z_0))
            
            return output
        
        
        valid = ~(np.isnan(gam) | np.isnan(alt_array))
        indice = len(gam[valid])//4 # the point used for the initial guess
        
        # Need at least 5 points to fit a curve
        if len(gam[valid]) >= 5:
            try:
                popt,pcov = curve_fit(func,alt_array[valid],gam[valid],
                            p0=[alt_array[valid][indice],1/7.,gam[valid][indice],90.])
                if i%15 ==0 or i==44 or i==20 or i==58:
                    plt.plot(alt_array,gam,'-o');plt.plot(alt_array[valid],func(alt_array[valid],*popt),'r',
                             label = '%.0f*exp(z - %.2f)/%.2f'%((popt[2]),(popt[0]),(1/popt[1]),)) 
                    this_time = dt.datetime.utcfromtimestamp(utime).strftime('%Y%m%d %H:%M')
                    plt.suptitle('fitstat %s'%(this_time)); plt.xlabel('Altitude (km)')
                    plt.ylabel('Gamma/HWHM (Hz)');plt.legend(); plt.show(); plt.close()
                    
                scale_height = 1/popt[1]
                
                popt_array[i , 0] = scale_height
                popt_array[i,1]   = popt[0]
                popt_array[i,2]   = popt[2]
                popt_array[i,3]   = popt[3]
            except RuntimeError:
                popt_array[i , 0] = np.nan
                popt_array[i,1]   = np.nan
                popt_array[i,2]   = np.nan
                popt_array[i,3]   = np.nan
        else:
            popt_array[i , 0] = np.nan
            popt_array[i,1]   = np.nan
            popt_array[i,2]   = np.nan
            popt_array[i,3]   = np.nan
    
    
       
    #### Convert our fitted parameters array to binary half hour binning data ###
    u =[dt.datetime.utcfromtimestamp(utime_array[i]) for i in range(len(utime_array))]
    uu = [datetime_bin(i) for i in u]
    uuu = sorted(set(uu))
    d= {el:[] for el in uuu}
    
    for i,key in enumerate(uu):
        d[key].append(popt_array[i,0])
    
    
    
    arr = np.zeros((len(uuu),2),dtype=object)
    for j,tup in enumerate(uuu):
        data = d[tup]
        good_data = [i for i in data if not np.isnan(i)]
        # if more than half contain scale height between good range
        percent = sum(i < 15 for i in good_data)/float(len(data))
        if percent >= 0.5:
            val = 1
        else:
            val=0
        
        arr[j,0] = tup
        arr[j,1] = val
    # save binary half hour binning data
    dirname  = join(basepath,'table_fit_data')
    if not exists(dirname):
        os.makedirs(dirname)
    fname = join(dirname,dname+'.npy')
    np.save(fname,arr)
    print 'done with %s'%(dname)
    return arr


## generate list of D-region experiments, lstt, to use fitstat ##
kk = sorted(os.listdir('/Volumes/scratch/AMISR_Work/spectra_Dregion'))
listt=[]
for i in kk:
    string = join('/Volumes/scratch/AMISR_Work/spectra_Dregion',i)
    if exists(join(string,'moments')):
        if len(os.listdir(join(string,'moments'))) > 5 and len(os.listdir(join(string,'datfiles'))) >=2:
            listt.append(string.split('/')[-1])



