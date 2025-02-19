import numpy as np 
from numba import njit
from scipy.optimize import curve_fit
from slow_nmr.slow_nmr.utils import traj_loader


@njit
def AutoCorr(q,corrdim,corrstride=1):
    """Autocorrelation function of input trajectory by brute force calculation.
    corrdim         length of correlation function
    corrstride      points to skip to get to longer times
    """

    ACF = np.zeros(corrdim)
    nsteps = q.shape[0]

    nsamples=nsteps/corrstride-np.linspace(0,nsteps/corrstride,nsteps/corrstride)
    nsamples=nsamples.astype(int)

    for i in range(0,corrdim):
        for j in range(0,nsamples[i]):
            ACF[i] += q[j]*q[j+i*corrstride]
    
    ACF = ACF/nsamples[0:corrdim]
    ACF = ACF-np.mean(q[::corrstride])**2;
    
    return ACF

def ACFfft(q,corrdim,corrstride=1):
    
    """Autocorrelation function of input trajectory.
    Unbiased autorcorrelation function by FFT
    corrdim         length of correlation function
    corrstride      points to skip to get to longer times
    Best performance if corrdim is a power of 2
    """
    
    f = q[::corrstride]
    
    N = f.shape[0]

    fvi = np.fft.fft(f, n=2*N)

    acf = np.real( np.fft.ifft( fvi * np.conjugate(fvi) )[:N] )

    acf = acf / (N - np.arange(N))
    
    acf = acf - np.mean(f)**2
    
    return acf[:corrdim]


# Exponential functions for fitting 
def oneexp0(t,tau):
    return np.exp(-t/tau)

def oneexp(t,a,tau):
    return a*np.exp(-t/tau)

def oneexp0off(t,tau,off):
    return (1.0-off)*np.exp(-t/tau)+off

def twoexp(t,a1,tau1,a2,tau2):
    return a1*np.exp(-t/tau1)+a2*np.exp(-t/tau2)

def threeexp(t,a1,tau1,a2,tau2,a3,tau3):
    return a1*np.exp(-t/tau1)+a2*np.exp(-t/tau2)+a3*np.exp(-t/tau3)

def fourexp(t,a1,tau1,a2,tau2,a3,tau3,a4,tau4):
    return a1*np.exp(-t/tau1)+a2*np.exp(-t/tau2)+a3*np.exp(-t/tau3)+a4*np.exp(-t/tau4)

def expprob(t,lam):
    return lam*np.exp(-t*lam)



def calc_acf(chem_shift, corrdim, corrstride):

    ob  = chem_shift - np.mean(chem_shift)

    ACF = ACFfft(ob,corrdim,corrstride)

    return ACF

def fit_acft(ACF, corrdim, dt, corrstride):

    tauaxis= np.linspace(0,corrdim*dt*corrstride,corrdim)
    
    initguess=np.array([ACF[0],4000,ACF[0],1000,50,ACF[0]]) # make later different with variables 

    # Assume some error
    Err=0.001*np.ones(len(tauaxis))                   

    # fit and store the parameters and the covariance matrix
    pars, _ = curve_fit(threeexp, tauaxis, ACF, p0=initguess, sigma=Err, maxfev=20000)

    #Now calculate the fitted data
    ACFfit = threeexp(tauaxis,pars[0],pars[1],pars[2],pars[3],pars[4],pars[5])
    #chisq2 = np.sum((ACF-ACFfit)**2)

    return ACFfit, pars