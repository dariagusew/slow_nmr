import numpy as np
from numba import njit

@njit
def force_object(q, spl_m, rc, rc_start):
    """ Evaluates the spline representation of the potential at x, returns the negativ
        gradient of the splines. Evaluates to value at the border beyond the border.  """

    width = np.mean(rc[1:] - rc[:-1])
    # find index of bin by shifting by the start of
    # the first bin and floor dividing
    idx = int((q - rc_start) // width)
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
def make_a_step(q_old, v_old, forces_old, dt, friction, spl_m, rc, rc_start, alpha, noise_scale):
    
    noise = np.random.normal(loc = 0, scale=1) 

    v_new = v_old * alpha + forces_old * (1-alpha)*friction + noise_scale * noise  
        
    q_new = q_old + v_new * dt 
    
    forces = force_object(q_new, spl_m, rc, rc_start) 

    return q_new, v_new, forces 

@njit 
def sim(n_steps, friction, dt, stride, q_init, spl_m, rc, rc_start, alpha, noise_scale):
        
    #define initial positions and velocities 
    q = q_init
    v_init = np.random.normal(loc=0,scale=1)
    v = v_init

    #define initial state 
    q_traj = np.zeros(n_steps // stride, dtype = np.float32)
    v_traj = np.zeros(n_steps // stride, dtype = np.float32)
    forces_traj = np.zeros(n_steps // stride, dtype = np.float32)

    forces = force_object(q, spl_m, rc, rc_start)

    for step in range(n_steps):

        q, v, forces = make_a_step(q, v, forces, dt, friction, spl_m, rc, rc_start, alpha, noise_scale)

        if step % stride == 0:
            q_traj[step // stride] = q
            v_traj[step // stride] = v
            forces_traj[step // stride] = forces

    return q_traj, v_traj, forces_traj