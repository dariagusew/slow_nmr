import numpy as np 
import pandas as pd
import mdtraj as md 
from numba import njit

def traj_loader(traj_path, dt, stride):
    
    traj  = np.load(traj_path)[::stride]
    time = np.linspace(0,len(traj)*dt*stride, len(traj))

    return time, traj 


def energy1d(traj, bins, T=298, R = 8.314*10**(-3)):
    
    hist, bins = np.histogram(traj,bins=bins,density=True)
    energy = -R*T*np.log(hist)
    energy = energy - np.min(energy)
    
    return bins, energy

def chem_shift_mean_maker(path_w, md_traj, md_top, ref_path,rc_path, res_idx, save = 'yes'):

    top = md.load(md_top)
    traj = md.load(md_traj,top=top)

    ref = np.load(ref_path)
    rc = np.genfromtxt(rc_path)
    
    fes_dt = ref[:,0][1] - ref[:,0][0]
    
    chem_shifts = pd.read_pickle(path_w)
    
    chem_shift_res = np.zeros(traj.n_frames)
    
    for frame in range(traj.n_frames):
        chem_shift_res[frame]=chem_shifts[frame][res_idx]['CA']
    
    idx = []
    for i in range(len(ref[:,0])):
        idx.append(np.where((rc[:,1]<ref[:,0][i]) & (rc[:,1]>=ref[:,0][i]-fes_dt)))
    
    w_mean_res = np.zeros(len(idx))
    
    for i in range(len(idx)):
        w_mean_res[i]=np.mean(chem_shift_res[idx[i]])

    w_mean_res = np.array(w_mean_res)
    
    if save=='yes':
        np.save(f'w_mean_res_{res_idx}', w_mean_res)

    return w_mean_res

def q_to_chem_shift_mapping(traj_path, ref_path, w_mean, dt, stride, save='no'):

    ref = np.load(ref_path)
    fes_dt = ref[:,0][1] - ref[:,0][0]

    _, traj = traj_loader(traj_path, dt, stride)
    
    idxq = []
    for i in range(len(ref[:,0])):
        idxq.append(np.where((traj<ref[:,0][i]) & (traj>=ref[:,0][i]-fes_dt)))
    
    w_mean = np.load(w_mean)
    
    chem_shift = np.zeros(traj.shape)
    
    for i in range(201):
        chem_shift[idxq[i]] = w_mean[i]
    
    if save=='yes':
        np.save(chem_shift,"chem_shift")
    
    return chem_shift