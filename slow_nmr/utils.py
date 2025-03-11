import numpy as np 
import pandas as pd
import mdtraj as md 
import os
import shutil
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

def chem_shift_mean_maker(path_w, md_traj, md_top, ref_path, rc_path, res_idx, atom='CA', save='yes',target_directory=None):
    # Load topology and trajectory data
    top = md.load(md_top)
    traj = md.load(md_traj, top=top)

    # Load reference and rc data
    ref = np.load(ref_path)
    rc = np.genfromtxt(rc_path)
    
    # Get the FES time delta
    fes_dt = ref[1, 0] - ref[0, 0]
    
    # Load chemical shifts
    chem_shifts = pd.read_pickle(path_w)
    
    # Initialize the array to store chemical shifts
    chem_shift_res = np.zeros(traj.n_frames)

    # Use a vectorized approach to extract chemical shifts for the specified residue index and atom
    for frame in range(traj.n_frames):
        chem_shift_res[frame] = chem_shifts.get(frame, {}).get(res_idx, {}).get(atom, np.nan)
    
    # Convert reference column to a NumPy array for faster access
    ref_times = ref[:, 0]
    
    # Create logical mask for rc values falling within the FES window
    idx = []
    for time in ref_times:
        idx.append((rc[:, 1] < time) & (rc[:, 1] >= (time - fes_dt)))
    
    # Convert list of masks to a NumPy array for faster processing
    idx_array = np.array(idx)
    
    # Calculate the weighted mean for each reference time
    w_mean_res = np.array([np.mean(chem_shift_res[idx_mask]) for idx_mask in idx_array])

    # Save the results if required
    if save == 'yes':
        output_file = f'w_mean_res_{res_idx}_{atom}.npy'
        np.save(output_file, w_mean_res)
        print(f"File saved as {output_file}")

        # If target_directory is provided, move the file to the specified directory
        if target_directory:
            # Ensure the target directory exists
            if not os.path.exists(target_directory):
                os.makedirs(target_directory)  # Create target directory if it does not exist
            
            # Move the file
            shutil.move(output_file, os.path.join(target_directory, os.path.basename(output_file)))
            print(f"File moved to {os.path.join(target_directory, os.path.basename(output_file))}")

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