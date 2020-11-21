#-------------------------------- EXAMPLE CODE (INCOMPLETE) --------------------------------------- 

import os
import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy import constants as phys_const
from scipy.fftpack import fft, ifft
import json

from frb_common import common_utils
from baseband_analysis.core import BBData
from baseband_analysis.utilities import incoherent_dedisp,get_main_peak_lim,get_spectrum_lim, get_snr, scrunch, upchannel

from RMutils.util_RM import do_rmsynth_planes,get_rmsf_planes,measure_FDF_parms,do_rmclean_hogbom
from baseband_analysis.do_QUfit_1D_mnest import run_qufit
from RMutils.util_misc import fit_spec_poly5, poly5

warnings.filterwarnings("ignore", category=RuntimeWarning) 

plt.rcParams["figure.figsize"] = [10, 10]
plt.rcParams.update({"font.size": 20})
            
            
def matched_filter(data_on,data_off,freq_id,tsmooth=128,fsmooth=None, statistic='mean', diagnostic_plots=False):

    """ 
    Takes waterfall intensity array (freq,time) and outputs array of weights used for integrating signal in time
    """

    I_on,_,_,_=get_stokes(
        data_on)
    
    I_off,_,_,_=get_stokes(
        data_off)
    
    I_on-=np.nanmean(I_off,-1)[...,np.newaxis]
    
    if fsmooth is None:

        weights1d=scrunch(I_on.sum(0),tscrunch=tsmooth,fscrunch=1, statistic=statistic)

        weights1d_hires=np.zeros(I_on.sum(0).size)
        fact=weights1d_hires.size/weights1d.size

        for i in range(weights1d_hires.size):
            idx=int(i/fact)
            weights1d_hires[i]=weights1d[idx]

        weights=weights1d_hires/weights1d_hires.sum()[...,np.newaxis]
        
        if diagnostic_plots:
            f, ax = plt.subplots(2,1,sharex=True)
            ax[0].plot(I_on.sum(0))
            ax[0].plot(weights1d_hires)
            ax[0].set_ylabel('Intensity')

            ax[1].plot(weights)
            ax[1].set_ylabel('weights')
            ax[1].set_xlabel('timesample')

    else:
        
        wfall=np.zeros((1024,I_on.shape[1]))
        wfall[:]=np.nan
        wfall[freq_id]=I_on
        
        wfall_s=scrunch(wfall,tscrunch=tsmooth,fscrunch=fsmooth, statistic=statistic)

        wfall_hires=np.zeros(wfall.shape)
        t_fact=wfall_hires.shape[1]/wfall_s.shape[1]
        f_fact=wfall_hires.shape[0]/wfall_s.shape[0]
    
        for i in range(wfall_hires.shape[0]):
            f_idx=int(i/f_fact)
            for j in range(wfall_hires.shape[1]):
                t_idx=int(j/t_fact)
                wfall_hires[i,j]=wfall_s[f_idx,t_idx]
            
            weights=wfall_hires/np.nansum(wfall_hires,1)[...,np.newaxis]
            weights=weights[freq_id]
            
        if diagnostic_plots:
            f, ax = plt.subplots(2,1,sharex=True)
            ax[0].plot(I_on.sum(0))
            ax[0].plot(wfall_hires.sum(0))
            ax[0].set_ylabel('Intensity')

            ax[1].imshow(weights, aspect='auto', origin='lower')
            ax[1].set_ylabel('Frequency [channel]')
            ax[1].set_xlabel('timesample')

    
    return weights



def plot_waterfall_raw(
    data, 
    freq_id, 
    time_range=None, 
    tscrunch=128, 
    fscrunch=8,
    diagnostic_plots=True):
    
    """Returns diagnostic plot of raw baseband intensity waterfall
    
    Parameters
    __________
    
    data : array_like
           formatted as (chan,pol,time)
    freq_id: vector
             frequency channel ids
    tscrunch : int                                                                                                              
               rebinning factor along first (e.g. time) axis                                                                     
    fscrunch : int                                                                                                              
               rebinning factor along second (e.g. frequency) axis 
    diagnostic_plots: bool or str
                      
                      
    Returns:
    ________
    Unbinned and rebinned intensity waterfall plots
    
    """
    if data.shape[1]==2: 
        power=abs(data[:,0,:])**2+abs(data[:,1,:])**2
    else: 
        log.info('processing input data as (chan,time) format')
        power=data.copy()
        
    nfreq=power.shape[0]
    bins=power.shape[-1]
    
    if (nfreq==common_utils.fpga_num_freq):
        wfall=power
    else:
        wfall=np.zeros((common_utils.fpga_num_freq, bins))
        wfall[freq_id,:]=power[:]

    for i in range(wfall.shape[0]):
        wfall[i,:]/=np.nanstd(wfall[i,:])

    mask=np.argwhere((common_utils.fpga_freq>730) & (common_utils.fpga_freq<760)) # LTE mask
    wfall[mask]=np.nan
    
    if time_range is not None:
        start_bin, end_bin = time_range
    else:
        start_bin, end_bin = 0, bins

    extent = [start_bin, end_bin, common_utils.freq_bottom_mhz, common_utils.freq_top_mhz]


    if diagnostic_plots:
        fig, ax = plt.subplots(dpi=100)
        plt.imshow(wfall, aspect='auto', origin='upper', extent=extent)
        plt.xlabel('time [bins]')
        plt.ylabel('Frequency [MHz]')
        if isinstance(diagnostic_plots, bool):
            plt.show()
        else:
            plot_name = "polarization_wfall_raw.png"
            plt.savefig(os.path.join(diagnostic_plots, plot_name))
            plt.close("all")
    
    wfall_s=scrunch(wfall,tscrunch=tscrunch,fscrunch=fscrunch)
    for i in range(wfall_s.shape[0]):
        wfall_s[i,:]/=np.nanmean(wfall_s[i,:])

    if diagnostic_plots:
        fig, ax = plt.subplots(dpi=100)
        plt.imshow(wfall_s, aspect='auto', origin='upper', extent=extent)
        plt.xlabel('time [bins]')
        plt.ylabel('Frequency [MHz]')
        if isinstance(diagnostic_plots, bool):
            plt.show()
        else:
            plot_name = "polarization_wfall_raw_rebin.png"
            plt.savefig(os.path.join(diagnostic_plots, plot_name))
            plt.close("all")


def get_baseband_burst(
    file_in, 
    DM=None, 
    time_range=None,
    width=None,
    power_out=False,
    diagnostic_plots=False):
    
    """Returns raw complex voltage array 
    
    Parameters
    __________
    
    beamformed_file: .h5 file
        single beam baseband file
    DM: float
        DM for event
    time_range: int tuple
    width: int
        width of output complex voltage array in FPGA sample units 
    diagnostic_plots: bool or str
    
    
    Returns
    _______
    
    bb: array_like
        raw voltage array (channels,pol,time)
    freq: vector
        channel frequencies [MHz]
    freq_id: vector
        channel ids
    optional: 2D waterfall plot
    """
    
    # Calculating baseband matrix
    data = BBData.from_file(file_in)

    # Refining the DM by maximizing S/N 
    DM_range = 10.
    power, offset, weight, valid_channels, dt, DM, downsampling_factor = get_calibrated_power(
        data,
        return_full=True,
        DM_range=DM_range,
        downsample=True,
    )
    bb, freq, freq_id = incoherent_dedisp(
        data,
        DM,
        fill_wfall = False
    )
    if time_range is None:
        time_range = dt

    freq = freq[valid_channels]
    freq_id = freq_id[valid_channels]
    bb = bb[valid_channels, :, time_range[0] : time_range[1]]

    peak_lim = np.array(get_main_peak_lim(
        power, 
        diagnostic_plots=diagnostic_plots, 
        normalize_profile=True, 
        floor_level=0.2)) * downsampling_factor

#     spect_lim = get_spectrum_lim(freq_id, power)
#     bb = bb[spect_lim[0] : spect_lim[1]]
#     freq = freq[spect_lim[0] : spect_lim[1]]
#     freq_id = freq_id[spect_lim[0] : spect_lim[1]]

    plot_waterfall_raw(
        bb,
        freq_id,
        diagnostic_plots=diagnostic_plots
    )
    
    if width is None:
        width=(peak_lim[1]-peak_lim[0])*5
        
    if power_out:
        
        if ((int(peak_lim.mean()-width) > 0) & (int(peak_lim.mean()+width) < bb.shape[-1])):
            power_cut=power[:,int(peak_lim.mean()-width):int(peak_lim.mean()+width)]
        else:
            print ('width exceeds maxmimum allowable value. resetting width to maximum possible value')
            width=np.min((int(peak_lim.mean()),int(bb.shape[-1]-peak_lim.mean())))
            power_cut=power[:,int(peak_lim.mean()-width):int(peak_lim.mean()+width)]
        
        return power, freq, freq_id
    
    else:
        
        if ((int(peak_lim.mean()-width) > 0) & (int(peak_lim.mean()+width) < bb.shape[-1])):
            bb_cut=bb[:,:,int(peak_lim.mean()-width):int(peak_lim.mean()+width)]
        else:
            print ('width exceeds maxmimum allowable value. resetting width to maximum possible value')
            width=np.min((int(peak_lim.mean()),int(bb.shape[-1]-peak_lim.mean())))
            bb_cut=bb[:,:,int(peak_lim.mean()-width):int(peak_lim.mean()+width)]
        
        return bb_cut, freq, freq_id 


def stokes_waterfall(data,  
                     freq_id, 
                     peak_lim,
                     weights=None, 
                     width=None, 
                     offset=0,
                     tscrunch=64,
                     fscrunch=4,
                     freq_lim=None,
                     diagnostic_plots=False):

    ### for producing diagnostic stokes waterfall plots ###
    
    '''
    In: baseband data formatted as (freq,pol,time),burst peak index, freq_id 
    
    Out: Stokes waterfall surrounding burst
    
    '''

    lim_lo, lim_hi = peak_lim
    if width is None:
        width=(lim_hi-lim_lo)*5,
    
    if ((int(peak_lim.mean()-width) > 0) & (int(peak_lim.mean()+width) < data.shape[-1])): 
        wfall_burst=data[:,:,int(peak_lim.mean()-width):int(peak_lim.mean()+width)]
    else: 
        width=np.min((int(peak_lim.mean()),int(data.shape[-1]-peak_lim.mean())))
        wfall_burst=data[:,:,int(peak_lim.mean()-width):int(peak_lim.mean()+width)]

    if weights is not None:
        wfall_burst[:,0,:]*=weights[...,np.newaxis]
        wfall_burst[:,1,:]*=weights[...,np.newaxis]

    bins=wfall_burst.shape[-1]
    npol=wfall_burst.shape[1]
    extent = [(-width)*2.56e-3, (width)*2.56e-3, common_utils.freq_bottom_mhz, common_utils.freq_top_mhz]
    
    wfall_full=np.zeros((common_utils.fpga_num_freq, npol, bins), dtype=np.complex64)
    
    wfall_full[freq_id,:,:]=wfall_burst[:]
    
    mask=np.argwhere((common_utils.fpga_freq>730) & (common_utils.fpga_freq<760))
    wfall_full[mask]=np.nan

    I,Q,U,V  = get_stokes(wfall_full)
    
    I_std=np.nanstd(I[:,:bins//4],axis=-1)
    Q_std=np.nanstd(Q[:,:bins//4],axis=-1)
    U_std=np.nanstd(U[:,:bins//4],axis=-1)
    V_std=np.nanstd(V[:,:bins//4],axis=-1)

    for i in range(I.shape[0]):
        I[i] -= np.nanmean(I[i,:bins//4])
        Q[i] -= np.nanmean(Q[i,:bins//4])
        U[i] -= np.nanmean(U[i,:bins//4])
        V[i] -= np.nanmean(V[i,:bins//4])
          
    I_s=I.copy()
    Q_s=Q.copy()
    U_s=U.copy()
    V_s=V.copy()

    for i in range(I.shape[0]):
        I_s[i,:]=I[i,:]/I_std[i]
        Q_s[i,:]=Q[i,:]/Q_std[i]
        U_s[i,:]=U[i,:]/U_std[i]
        V_s[i,:]=V[i,:]/V_std[i]
        
    I_s=scrunch(I_s, tscrunch, fscrunch)
    Q_s=scrunch(Q_s, tscrunch, fscrunch)
    U_s=scrunch(U_s, tscrunch, fscrunch)
    V_s=scrunch(V_s, tscrunch, fscrunch)
    
    if freq_lim is None:
        freq_lim=[400,800]
        
    if diagnostic_plots:
        lim_lo=lim_lo-peak_lim.mean()
        lim_hi=lim_hi-peak_lim.mean()
        
        fig, ax = plt.subplots(2,2, figsize=(10,10), dpi=250)#, sharex=True, sharey=True)
        plt.subplots_adjust(left=0.1, bottom=0.15, right=0.99, top=0.90, wspace=0, hspace=0)

        ax[0,0].imshow(I_s, aspect='auto', origin='upper', extent=extent)
        ax[0,0].plot(np.repeat((lim_lo+offset)*2.56e-3,1000), np.linspace(400,800,1000), ls=':', color='r')
        ax[0,0].plot(np.repeat((lim_hi+offset)*2.56e-3,1000), np.linspace(400,800,1000), ls=':', color='r')
        ax[0,0].set_xticks([],[])
        ax[0,0].set_ylabel('Frequency [MHz]')
        ax[0,0].set_ylim([freq_lim[0],freq_lim[1]])
        ax[0,0].set_title('Stokes I')

        ax[0,1].imshow(V_s, aspect='auto', origin='upper', extent=extent)
        ax[0,1].plot(np.repeat((lim_lo+offset)*2.56e-3,1000), np.linspace(400,800,1000), ls=':', color='r')
        ax[0,1].plot(np.repeat((lim_hi+offset)*2.56e-3,1000), np.linspace(400,800,1000), ls=':', color='r')
        ax[0,1].set_yticks([],[])
        ax[0,1].set_xticks([],[])
        ax[0,1].set_ylim([freq_lim[0],freq_lim[1]])
        ax[0,1].set_title('Stokes V')
       
        ax[1,0].imshow(Q_s, aspect='auto', origin='upper', extent=extent)
        ax[1,0].plot(np.repeat((lim_lo+offset)*2.56e-3,1000), np.linspace(400,800,1000), ls=':', color='r')
        ax[1,0].plot(np.repeat((lim_hi+offset)*2.56e-3,1000), np.linspace(400,800,1000), ls=':', color='r')
        ax[1,0].set_xlabel('Time [ms]')
        ax[1,0].set_ylabel('Frequency [MHz]')
        ax[1,0].set_ylim([freq_lim[0],freq_lim[1]])
        ax[1,0].set_title('Stokes Q')
     
        ax[1,1].imshow(U_s, aspect='auto', origin='upper', extent=extent)
        ax[1,1].plot(np.repeat((lim_lo+offset)*2.56e-3,1000), np.linspace(400,800,1000), ls=':', color='r')
        ax[1,1].plot(np.repeat((lim_hi+offset)*2.56e-3,1000), np.linspace(400,800,1000), ls=':', color='r')
        ax[1,1].set_xlabel('Time [ms]')
        ax[1,1].set_yticks([],[])
        ax[1,1].set_ylim([freq_lim[0],freq_lim[1]])
        ax[1,1].set_title('Stokes U')
        
        plt.tight_layout()
        
        if isinstance(diagnostic_plots, bool):
            plt.show()
        else:
            plot_name = "stokes_waterfall.pdf"
            plt.savefig(os.path.join(diagnostic_plots, plot_name))
            plt.close("all")

def get_stokes(
    data):
    
    X=data[:,0,:]
    Y=data[:,1,:]

    I=abs(X)**2+abs(Y)**2
    Q=abs(X)**2-abs(Y)**2
    U=2*np.real(X*np.conj(Y))
    V=-2*np.imag(X*np.conj(Y))

    return I,Q,U,V


def extract_stokes(
    data, 
    freq,
    stokes_off=None,
    weights=None, 
    diagnostic_plots=False):
        
    I,Q,U,V=get_stokes(
        data)
    
    nbins=I.shape[-1]
    
    if weights is None:
        I_on=np.nanmean(I,-1)
        Q_on=np.nanmean(Q,-1)
        U_on=np.nanmean(U,-1)
        V_on=np.nanmean(V,-1)
    else:
        I_on=np.sum(I*weights,-1)/np.sum(weights,-1) 
        Q_on=np.sum(Q*weights,-1)/np.sum(weights,-1) 
        U_on=np.sum(U*weights,-1)/np.sum(weights,-1) 
        V_on=np.sum(V*weights,-1)/np.sum(weights,-1) 
        
    if stokes_off is not None:        
        freq,I_off,Q_off,U_off,V_off,I_std,Q_std,U_std,V_std=stokes_off    
        
        I_on-=I_off
        Q_on-=Q_off
        U_on-=U_off
        V_on-=V_off
        
        # re-scales the channel noise by number of bins in on-pulse region (uncertainty in average Stokes)
        I_std /= np.sqrt(nbins)
        Q_std /= np.sqrt(nbins)
        U_std /= np.sqrt(nbins)
        V_std /= np.sqrt(nbins)

    else:
        I_std=np.nanstd(I,axis=-1)
        Q_std=np.nanstd(Q,axis=-1)
        U_std=np.nanstd(U,axis=-1)
        V_std=np.nanstd(V,axis=-1)
    
    lam=phys_const.speed_of_light/freq/1e6
    lam2=lam**2
        
    if diagnostic_plots:
        
        fig, ax = plt.subplots(4,1, figsize=(20,15), sharex=True)

        for axx in ax:
            axx.tick_params(labelsize=20)

        ax[0].scatter(freq,I_on/I_std)
        ax[0].set_title('Stokes I S/N ', fontsize=20)
        ax[0].set_ylim([0,5*np.nanmedian(abs(I_on/I_std))])
        ax[1].scatter(freq,Q_on/Q_std)
        ax[1].set_title('Stokes Q S/N', fontsize=20)
        ax[1].set_ylim([-5*np.nanmedian(abs(Q_on/Q_std)),5*np.nanmedian(abs(Q_on/Q_std))])
        ax[2].scatter(freq,U_on/U_std)
        ax[2].set_title('Stokes U S/N', fontsize=20)
        ax[2].set_ylim([-5*np.nanmedian(abs(U_on/U_std)),5*np.nanmedian(abs(U_on/U_std))])
        ax[3].scatter(freq,V_on/V_std)
        ax[3].set_title('Stokes V S/N', fontsize=20)
        ax[3].set_xlabel('Freq. [MHz]', fontsize=20)

        plt.tight_layout()
        
        if isinstance(diagnostic_plots, bool):
            plt.show()
        else:
            plot_name = "stokes_vs_freq.png"
            plt.savefig(os.path.join(diagnostic_plots, plot_name))
            plt.close("all")
        
        fig, ax = plt.subplots(4,1, figsize=(20,15), sharex=True)

        for axx in ax:
            axx.tick_params(labelsize=20)

        ax[0].scatter(lam2,I_on/I_std)
        ax[0].set_title('Stokes I S/N ', fontsize=20)
        ax[0].set_ylim([0,5*np.nanmedian(abs(I_on/I_std))])
        ax[1].scatter(lam2,Q_on/Q_std)
        ax[1].set_title('Stokes Q S/N', fontsize=20)
        ax[1].set_ylim([-5*np.nanmedian(abs(Q_on/Q_std)),5*np.nanmedian(abs(Q_on/Q_std))])
        ax[2].scatter(lam2,U_on/U_std)
        ax[2].set_title('Stokes U S/N', fontsize=20)
        ax[2].set_ylim([-5*np.nanmedian(abs(U_on/U_std)),5*np.nanmedian(abs(U_on/U_std))])
        ax[3].scatter(lam2,V_on/V_std)
        ax[3].set_title('Stokes V S/N', fontsize=20)
        ax[3].set_xlabel('$\lambda$ [m$^2$]', fontsize=20)

        plt.tight_layout()
        
        if isinstance(diagnostic_plots, bool):
            plt.show()
        else:
            plot_name = "stokes_vs_lam2.png"
            plt.savefig(os.path.join(diagnostic_plots, plot_name))
            plt.close("all")
    
    return freq,I_on,Q_on,U_on,V_on,I_std,Q_std,U_std,V_std


def RM_synth(stokes,
             weight=True, # weighting needs more testing. Fails on certain events
             upchan=False, 
             RM_lim=None, 
             nSamples=None,
             normed=False,
             noise_type='theory',
             diagnostic_plots=False,
             cutoff=None):
    
    """ Performs rotation measure synthesis to extract Faraday Dispersion Function and RM measurement
    
    Parameters
    __________
    
    stokes : list (freq,I,Q,U,V,dI,dQ,dU,dV)
        Stokes params.
    weight : Bool. 
        Freq. channel std., used as to create array of weights for FDF
    band_lo, band_hi : float
        bottom and top freq. band limits of burst. 
    upchan: Bool
        If true, sets nSamples=3 
    RM_lim: float (optional)
        limits in Phi space to calculate the FDF
    nSamples: int (optional)
        sampling density in Phi space
    diagnostic_plots: Bool.
        outputs diagnostic plots of FDF
            
    Returns
    _______
    FDF params, (phi, FDF_arr) : list, array_like
        (RM,RM_err,PA,PA_err), FDF_arr
    """

    freqArr=stokes[0].copy()
    IArr=stokes[1].copy()
    QArr=stokes[2].copy()
    UArr=stokes[3].copy()
    VArr=stokes[4].copy()
    dIArr=stokes[5].copy()
    dQArr=stokes[6].copy()
    dUArr=stokes[7].copy()
    dVArr=stokes[8].copy()


    freqArr_Hz=freqArr*1e6
    lamArr_m=phys_const.speed_of_light/freqArr_Hz # convert to wavelength in m
    lambdaSqArr_m2=lamArr_m**2
    
    if normed is True:
        dQArr=IArr*np.sqrt((dQArr/QArr)**2+(dIArr/IArr)**2)
        dUArr=IArr*np.sqrt((dUArr/UArr)**2+(dIArr/IArr)**2)
        QArr/=IArr
        UArr/=IArr


    dQUArr = (dQArr + dUArr)/2.0
    if weight is True:
        weightArr = 1.0 / np.power(dQUArr, 2.0)
    else:
        weightArr = np.ones(freqArr_Hz.shape, dtype=float)
    dFDFth = np.sqrt( np.sum(weightArr**2 * dQUArr**2) / (np.sum(weightArr))**2 ) # check this equation!!!
    
    if nSamples is None:
        if upchan:
            nSamples=3 # sampling resolution of the FDF. 
        else:
            nSamples=10

    lambdaSqRange_m2 = (np.nanmax(lambdaSqArr_m2) - np.nanmin(lambdaSqArr_m2) )  
    fwhmRMSF_radm2 = 2.0 * np.sqrt(3.0) / lambdaSqRange_m2

#     dLambdaSqMin_m2 = np.nanmin(np.abs(np.diff(lambdaSqArr_m2)))
#     dLambdaSqMax_m2 = np.nanmax(np.abs(np.diff(lambdaSqArr_m2)))
    dLambdaSqMed_m2 = np.nanmedian(np.abs(np.diff(lambdaSqArr_m2)))

    dPhi_radm2 = fwhmRMSF_radm2 / nSamples

    #     phiMax_radm2 = np.sqrt(3.0) / dLambdaSqMax_m2
    phiMax_radm2 = np.sqrt(3.0) / dLambdaSqMed_m2 # sets the RM limit that can be probed based on intrachannel depolarization
    phiMax_radm2 = max(phiMax_radm2,600)    # Force the minimum phiMax

    if RM_lim  is None:
        # Faraday depth sampling. Zero always centred on middle channel
        nChanRM = int(round(abs((phiMax_radm2 - 0.0) / dPhi_radm2)) * 2.0 + 1.0)
        startPhi_radm2 = - (nChanRM-1.0) * dPhi_radm2 / 2.0
        stopPhi_radm2 = + (nChanRM-1.0) * dPhi_radm2 / 2.0
        phiArr_radm2 = np.linspace(startPhi_radm2, stopPhi_radm2, nChanRM)

    else:
        startPhi_radm2 = RM_lim[0]
        stopPhi_radm2 = RM_lim[1]
        nChanRM = int(round(abs((((stopPhi_radm2-startPhi_radm2)//2) - 0.0) / dPhi_radm2)) * 2.0 + 1.0)
        phiArr_radm2 = np.linspace(startPhi_radm2, stopPhi_radm2, nChanRM)
    
    phiArr_radm2 = phiArr_radm2.astype(np.float)
  
    ### constructing FDF ###

    dirtyFDF, lam0Sq_m2 = do_rmsynth_planes(
    QArr, 
    UArr, 
    lambdaSqArr_m2, 
    phiArr_radm2)

    RMSFArr, phi2Arr_radm2, fwhmRMSFArr, fitStatArr = get_rmsf_planes(
    lambdaSqArr_m2 = lambdaSqArr_m2,
    phiArr_radm2 = phiArr_radm2,
    weightArr=weightArr, 
    mskArr=None,
    lam0Sq_m2=lam0Sq_m2, 
    double = True) # routine needed for RM-cleaning 

    FDF, lam0Sq_m2 = do_rmsynth_planes(
    QArr, 
    UArr, 
    lambdaSqArr_m2, 
    phiArr_radm2,                   
    weightArr=weightArr, 
    lam0Sq_m2=None,
    nBits=32, 
    verbose=False)

    FDF_max=np.argmax(abs(FDF))
    FDF_med=np.median(abs(FDF))

    dFDFobs=np.median(abs(abs(FDF)-FDF_med)) / np.sqrt(np.pi/2) #MADFM definition of noise
#   dFDF_obs=np.nanstd(abs(FDF)) #std. definition of noise

    if noise_type is 'observed':
        FDF_snr=abs((abs(FDF)-FDF_med)/dFDFobs)/2
    if noise_type is 'theory':
        FDF_snr=abs((abs(FDF)-FDF_med)/dFDFth)/2

    mDict = measure_FDF_parms(FDF         = dirtyFDF,
                              phiArr      = phiArr_radm2,
                              fwhmRMSF    = fwhmRMSF_radm2,
                              dFDF        = dFDFth, #FDF_noise
                              lamSqArr_m2 = lambdaSqArr_m2,
                              lam0Sq      = lam0Sq_m2)

    RM_radm2_fit=mDict["phiPeakPIfit_rm2"]
    dRM_radm2_fit=mDict["dPhiPeakPIfit_rm2"]

    RM_radm2=phiArr_radm2[FDF_max]
    dRM_radm2=fwhmRMSF_radm2/(2*FDF_snr.max())

    polAngle0Fit_deg=mDict["polAngle0Fit_deg"]
    dPolAngle0Fit_deg=mDict["dPolAngle0Fit_deg"] * np.sqrt(freqArr.size) # np.sqrt(freqArr.size) term corrects for band-average noise 

    
    if cutoff is None:
    
        if diagnostic_plots:
            fig, ax = plt.subplots(2,1, figsize=(20,10))
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.99, top=0.95, wspace=0)
            ax[0].set_title('Faraday Dispersion Function')    
            ax[0].plot(phiArr_radm2,FDF_snr)
            ax[0].set_xlim([phiArr_radm2.min(), phiArr_radm2.max()])
            ax[1].plot(phiArr_radm2,FDF_snr)
            #     ax[1].axvline(RM_radm2, color='k', ls=':', label=r'RM=%.2f $\pm$ %0.2f rad/m$^2$' %(RM_radm2,dRM_radm2))
            ax[1].set_xlim(phiArr_radm2[FDF_max]-300,phiArr_radm2[FDF_max]+300)
            ax[1].set_xlabel('$\phi$ [rad/m$^2$]')
            fig.text(0.03, 0.5, 'Polarized Intensity [S/N]', va='center', rotation='vertical')
            #     plt.legend(fontsize=20)
            #     plt.tight_layout()

            if isinstance(diagnostic_plots, bool):
                plt.show()
            else:
                plot_name = "FDF.png"
                plt.savefig(os.path.join(diagnostic_plots, plot_name))
                plt.close("all")
            
        return (RM_radm2_fit,RM_radm2,dRM_radm2_fit,dRM_radm2,polAngle0Fit_deg,dPolAngle0Fit_deg),(phiArr_radm2,FDF_snr)

    else:

        if noise_type is 'observed':
            cutoff_abs = dFDFobs*cutoff
        if noise_type is 'theory':
            cutoff_abs = dFDFth*cutoff

        cleanFDF, ccArr, iterCountArr, residFDF = do_rmclean_hogbom(dirtyFDF = FDF,
                                phiArr_radm2    = phiArr_radm2,
                                RMSFArr         = RMSFArr,
                                phi2Arr_radm2   = phi2Arr_radm2,
                                fwhmRMSFArr     = fwhmRMSF_radm2,
                                cutoff          = cutoff_abs,
    #                                 maxIter         = maxIter,
    #                                 gain            = gain,
    #                                 verbose         = verbose,
                                doPlots         = True)


        FDF_max=np.argmax(abs(cleanFDF))
        FDF_med=np.median(abs(cleanFDF))

        dFDFobs=np.median(abs(abs(cleanFDF)-FDF_med)) / np.sqrt(np.pi/2) #MADFM definition of noise                                                      #   dFDF_obs=np.nanstd(abs(FDF)) #std. definition of noise                                                                                      
        if noise_type is 'observed':
            FDF_snr_clean=abs((abs(cleanFDF)-FDF_med)/dFDFobs)/2
            ccArr_snr=(abs(ccArr)/dFDFobs)/2
        if noise_type is 'theory':
            FDF_snr_clean=abs((abs(cleanFDF)-FDF_med)/dFDFth)/2
            ccArr_snr=(abs(ccArr)/dFDFth)/2

        mDict = measure_FDF_parms(FDF     = cleanFDF,
                              phiArr      = phiArr_radm2,
                              fwhmRMSF    = fwhmRMSF_radm2,
                              dFDF        = dFDFth, #FDF_noise
                              lamSqArr_m2 = lambdaSqArr_m2,
                              lam0Sq      = lam0Sq_m2)

        RM_radm2_fit=mDict["phiPeakPIfit_rm2"]
        dRM_radm2_fit=mDict["dPhiPeakPIfit_rm2"]

        RM_radm2=phiArr_radm2[FDF_max]
        dRM_radm2=fwhmRMSF_radm2/(2*FDF_snr_clean.max())

        polAngle0Fit_deg=mDict["polAngle0Fit_deg"]
        dPolAngle0Fit_deg=mDict["dPolAngle0Fit_deg"] * np.sqrt(freqArr.size) # np.sqrt(freqArr.size) term corrects for band-average noise 


        if diagnostic_plots:
            fig, ax = plt.subplots(2,1, figsize=(20,10))
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.99, top=0.95, wspace=0)
            ax[0].set_title('Faraday Dispersion Function')    
            ax[0].plot(phiArr_radm2,FDF_snr_clean, label='clean FDF')
            ax[0].plot(phiArr_radm2,FDF_snr, label='dirty FDF')
            ax[0].axhline(cutoff, ls='--', color='k', label='clean cutoff')
            ax[0].legend()
            ax[0].set_xlim([phiArr_radm2.min(), phiArr_radm2.max()])
            ax[1].plot(phiArr_radm2,FDF_snr_clean, label='clean FDF')
            ax[1].plot(phiArr_radm2,FDF_snr, label='dirty FDF')
            ax[1].axhline(cutoff, ls='--', color='k',label='clean cutoff')
            ax[1].legend()
            ax[1].set_xlim(phiArr_radm2[FDF_max]-300,phiArr_radm2[FDF_max]+300)
            ax[1].set_xlabel('$\phi$ [rad/m$^2$]')
            fig.text(0.03, 0.5, 'Polarized Intensity [S/N]', va='center', rotation='vertical')

            if isinstance(diagnostic_plots, bool):
                plt.show()
            else:
                plot_name = "FDF.png"
                plt.savefig(os.path.join(diagnostic_plots, plot_name))
                plt.close("all")
                            
            fig, ax = plt.subplots(2,1,figsize=(20,10))
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.99, top=0.95, wspace=0, hspace=0.0)
            ax[0].set_title('Faraday Dispersion Function')    
            ax[0].plot(phi2Arr_radm2,RMSFArr, label='RMTF')
            ax[0].set_xlim([-300,300])
            ax[0].xaxis.set_ticklabels([])
            ax[0].legend()
            ax[1].plot(phiArr_radm2,FDF_snr_clean, label='clean FDF')
            ax[1].bar(phiArr_radm2,ccArr_snr, color='g', label='clean components')
            ax[1].legend()
            ax[1].set_xlim(phiArr_radm2[FDF_max]-300,phiArr_radm2[FDF_max]+300)
            ax[1].set_xlabel('$\phi$ [rad/m$^2$]')
            fig.text(0.03, 0.5, 'Polarized Intensity [S/N]', va='center', rotation='vertical')
            #     plt.legend(fontsize=20)
            #     plt.tight_layout()

            if isinstance(diagnostic_plots, bool): 
                plt.show()
            else:
                plot_name = "FDF_clean.png"
                plt.savefig(os.path.join(diagnostic_plots, plot_name))
                plt.close("all")

        return (RM_radm2_fit,RM_radm2,dRM_radm2_fit,dRM_radm2,polAngle0Fit_deg,dPolAngle0Fit_deg),(phiArr_radm2,FDF_snr_clean)



\
        'PA_FDF' : PA_FDF,
        'PA_err_QUfit': PA_err_QUfit,
        'PA_err_FDF' : PA_FDF_err,
        'RM_QUfit' : RM_QUfit,
        'RM_FDF_fit' : RM_FDF_fit,
        'RM_FDF' : RM_FDF,
        'RM_err_QUfit' : RM_err_QUfit,
        'RM_err_FDF_fit': RM_FDF_err_fit,
        'RM_err_FDF': RM_FDF_err,
        'cable_delay': lag,
        'cable_delay_err': lag_err
    }


def semicoherent_search(bb_on, bb_off, freq, RM_range=1e6, RM_step=None, diagnostic_plots=False, SNR_lim=6):
    """Performs a semi-coherent search over a range of Rotation Measure"""
    # Define the RM search
    if RM_step is None:
        RM_step = get_RM_step()
    RM_list = np.vstack(
        [
            np.arange(RM_step, RM_range+RM_step, RM_step),
            - np.arange(RM_step, RM_range+RM_step, RM_step)
        ]
    ).T.flatten()

    # Perform the RM search
    for i,RM in enumerate(RM_list):
        sys.stdout.write("\r")
        sys.stdout.write("Completed {:.1f}%".format(float(i) / RM_list.size * 100))
        sys.stdout.flush()
        
        bb_on_derot = coherent_derotation(bb_on.copy(), RM, freq)
        bb_off_derot = coherent_derotation(bb_off.copy(), RM, freq)

        # Extract off pulse spectrum
        stokes_off = extract_off_pulse(
            bb_off_derot,
            freq,
        )
        stokes_on = extract_on_pulse(
            bb_on_derot,
            freq,
            stokes_off=stokes_off,
        )
        params, (phi, FDF_arr) = RM_synth(
            stokes_on,
            weight=False,
            diagnostic_plots=False,
            RM_lim=[- RM_step / 2., RM_step / 2.],
            nSamples=3
        )

        if FDF_arr.max() > SNR_lim:
            break
    sys.stdout.write("\n")
    sys.stdout.flush()

    if FDF_arr.max() < SNR_lim:
        return None
    
    if diagnostic_plots:
        plt.plot(phi + RM, FDF_arr, 'k')
        #plt.axvline(RM, color='g')
        plt.xlabel("RM [rad/m2]")
        plt.ylabel("S/N")
        plt.tight_layout()
        if isinstance(diagnostic_plots, str) or isinstance(diagnostic_plots, unicode):
            plot_name = "RM_FDF.png"
            plt.savefig(os.path.join(diagnostic_plots, plot_name))
            plt.close("all")
        else:
            plt.show()

    RM_shift, RM_err, PA, PA_err = params
    RM += RM_shift
    return RM, RM_err, PA, PA_err
    
