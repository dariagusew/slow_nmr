import numpy as np
from numba import njit
import scipy
from numba_progress import ProgressBar
from tqdm import tqdm

@njit
def force_object(q, spl_m, rc, rc_start):
    """ Evaluates the spline representation of the potential at x, returns the negativ
        gradient of the splines. Evaluates to value at the border beyond the border.  """

    width = np.mean(rc[1:] - rc[:-1])
    # find index of bin by shifting by the start of
    # the first bin and floor dividing
    idx = int((q - rc_start) // width)
    print(idx)
    # set gradient of the free energy beyond the borders
    # to the value at the border
    if idx < 0:
        idx = 0
        q = rc_start
    elif idx > len(rc) - 2:
        idx = len(rc) - 2
        q = rc[0] + (len(rc) - 1) * width
    # evaluate the gradient of the spline rep
    output = -(
        3 * spl_m[idx, 0] * (q - rc[idx]) ** 2
        + 2 * spl_m[idx, 1] * (q - rc[idx])
        + spl_m[idx, 2]
    )
    return output


@njit
def sim(q_init, friction, masses, beta, dt, n_steps, stride, spl_m, rc, rc_start, sigv=1):
        
    #define initial positions and velocities 
    q = q_init

    v_init = np.random.normal(loc=0,scale=1)
    v = v_init

    alpha = np.exp(-1*friction * dt) 
    noise_scale = np.sqrt((1-alpha*alpha)*masses/beta)

    noise = np.random.normal(loc = 0, scale = sigv) 
    
    #define initial state 
    q_traj = np.zeros(n_steps // stride)
    v_traj = np.zeros(n_steps // stride)
    forces_traj = np.zeros(n_steps // stride)

        
    #collision rate is pjump/delt
    pjump = 0.025
    ncoll = 0
    cut = np.exp(-pjump)

    #forces = force_object(q, spl_m)
    forces = force_object(q, spl_m, rc, rc_start)

    for step in tqdm(range(n_steps), desc = 'simulation timestep'):
        
        expdist = np.random.random(n_steps) > cut

        v_new = v * alpha + forces * (1-alpha) * friction + noise_scale * noise  
        
        q_new = q + v_new * dt 
    
        #forces_new = force_object(q_new, spl_m)
        forces_new = force_object(q_new, spl_m, rc, rc_start)

        q = q_new
        v = v_new
        forces = forces_new
        
        #q, v, forces, potential = self.integrator.make_a_step(q, v, forces)
            
        # Test for a collision occurance
        if expdist[step]:
            ncoll = ncoll+1
            v = np.random.normal(loc=0,scale=sigv)

        if step % stride == 0:
            q_traj[(step - 1) // stride] = q_new
            v_traj[(step - 1) // stride] = v_new
            forces_traj[(step - 1) // stride] = forces_new
        
        np.save('q_traj', q_traj)
        np.save('v_traj', v_traj)
        np.save('f_traj', forces_traj)

    return q_traj, v_traj, forces_traj