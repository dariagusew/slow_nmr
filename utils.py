import numpy as np 

def traj_loader(traj_path, dt, stride):
    
    traj  = np.load(traj_path[::stride])
    time = np.linspace(0,len(traj)*dt*stride, len(traj))

    return time, traj 

def energy1d(traj, bins, T=298, R = 8.314*10**(-3)):
    
    hist, bins = np.histogram(traj,bins=bins,density=True)
    energy = -R*T*np.log(hist)
    energy = energy - np.min(energy)
    
    return bins, energy
