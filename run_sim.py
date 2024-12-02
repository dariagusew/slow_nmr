from jsonargparse import CLI
import numpy as np
import scipy
from simulation import sim
from time import ctime

def run_sim(n_steps: int,
            stride: int,  
            fes_path: str,
            q_init: int,
            masses: int,
            dt: float,
            friction: float,
            beta: float
            ):
    
    fes = np.load(fes_path)
    rc = fes[:,0]
    rc_start = rc[0] 
    pot = fes[:,1]
    cs = scipy.interpolate.CubicSpline(rc, pot)
    spl_m = cs.c.T

    alpha = np.exp(-1*friction * dt) 
    noise_scale = np.sqrt((1-alpha*alpha)*masses/(beta))

   
    q_traj, v_traj, f_traj = sim(q_init=q_init,
                                 friction=friction,
                                 dt=dt,
                                 n_steps=n_steps,
                                 stride=stride,
                                 spl_m=spl_m,
                                 rc=rc,
                                 rc_start=rc_start,
                                 alpha = alpha, 
                                 noise_scale= noise_scale)

    np.save('q_traj', q_traj)
    np.save('v_traj', v_traj)
    np.save('f_traj', f_traj)

if __name__ == "__main__":
    print("Start run_sim.py: {}".format(ctime()))
    
    CLI(run_sim)

    print("Finish run_sim.py: {}".format(ctime()))
