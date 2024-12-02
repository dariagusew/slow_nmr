import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','no-latex'])
from .utils import traj_loader, energy1d,  q_to_chem_shift_mapping, calc_acf


def plot_traj(traj_path,dt, stride, save_fig ='no'):
    time, traj  = traj_loader(traj_path, dt, stride)

    plt.figure(figsize=(4,2),dpi=300)
    plt.plot(time, traj)
    plt.xlabel(r'$t/10^7$',fontsize=15)
    plt.ylabel(r'$S_{path}$',fontsize=15)
    if save_fig == 'yes':
        plt.savefig('traj_path.pdf',dpi=300)
    plt.show()

def plot_chem_shift(traj_path,ref_path, dt, w_mean_path, stride,n_steps,  save_fig ='no'):
    
    chem_shift = q_to_chem_shift_mapping(traj_path, ref_path, w_mean_path, dt, stride)
    time = np.linspace(0, n_steps*dt*stride, n_steps)

    plt.figure(figsize=(4,2),dpi=300)
    plt.plot(time, chem_shift)
    plt.xlabel(r'$t$',fontsize=15)
    plt.ylabel(r'$<\delta> [ppm]$',fontsize=15)
    if save_fig == 'yes':
        plt.savefig('chem_shift_path.pdf',dpi=300)
    plt.show()

def plot_energy(ref_path, traj_path, dt, stride, save_fig='no'):

    ref = np.load(ref_path)
    rc = ref[:,0]
    fes = ref[:,1]
    _, traj = traj_loader(traj_path, dt, stride)
    bins, energy = energy1d(traj, bins=100)
    

    plt.figure(dpi=300)
    plt.plot(rc,(fes-np.min(fes)),c='k',label= 'MD')
    plt.plot(bins[1:],energy[1],label= 'LD')
    plt.xlabel('S$_{path}$',fontsize=15)
    plt.ylabel('Free energy [kJ/mol]',fontsize=15)
    plt.legend(frameon = True, bbox_to_anchor=(1, 0, 0, 1))
    if save_fig == 'yes':
        plt.savefig('energy.pdf',dpi=300)
    plt.show()

def plot_acf(ob_idx, traj_path, chem_shift_path, dt, stride, corrdim, corrstride, save_fig ='no'):

    tauaxis, ACF = calc_acf(ob_idx, traj_path, chem_shift_path, dt, stride, corrdim, corrstride)

    plt.figure(dpi=300)
    plt.plot(tauaxis, ACF)
    plt.ylabel(r'$C(\tau)$',fontsize=15)
    plt.xlabel(r'$\tau$',fontsize=15)
    if save_fig == 'yes':
        plt.savefig('acf.pdf',dpi=300)
    plt.show()