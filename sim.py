import numpy as np 
from numba import njit 
from tqdm import tqdm
from scipy.interpolate import CubicSpline

#def forces_from_fes(q, path):
    
#    fes = np.load(path)

#    rc = fes[:,0]
#    pot = fes[:,1]

#    cs = CubicSpline(rc, pot)

#    potential_on_q  = cs(q)
#    forces_on_q = -cs(q, 1)

#    return forces_on_q, potential_on_q 
@njit
def du_dx(q, spl_m, values, q_init, width):
    """ Evaluates the spline representation of the potential at x, returns the negativ
        gradient of the splines. Evaluates to value at the border beyond the border.  """

    # find index of bin by shifting by the start of
    # the first bin and floor dividing
    idx = int((q - q_init) // width)
    # set gradient of the free energy beyond the borders
    # to the value at the border
    if idx < 0:
        idx = 0
        q = q_init
    elif idx > len(values) - 2:
        idx = len(values) - 2
        q = values[0] + (len(values) - 1) * width
    # evaluate the gradient of the spline rep
    output = -(
        3 * spl_m[idx, 0] * (q - values[idx]) ** 2
        + 2 * spl_m[idx, 1] * (q - values[idx])
        + spl_m[idx, 2]
    )
    return output


    #def make_a_step(q_old, v_old, forces_old, friction, masses, beta, dt, spl_m, values, q_init, width):

#    alpha = np.exp(-1*friction * dt) 
#    noise_scale = np.sqrt((1-alpha*alpha)*(masses)/(beta))

#    noise = np.random.normal(loc = 0, scale=1) 

#    v_new = v_old * alpha + forces_old * (1-alpha) * friction + noise_scale * noise  

#    q_new = q_old + v_new * dt 

#    forces = du_dx(q_new, spl_m, values, q_init, width)

 #   return q_new, v_new, forces 

@njit 
def run_sim(q_init, friction, masses, beta, dt, n_steps, stride, spl_m, values, width, sigv=1):
        
    #define initial positions and velocities 
    q = q_init
    v_init = np.random.normal(loc=0,scale=1)
    v = v_init

    alpha = np.exp(-1*friction * dt) 
    noise_scale = np.sqrt((1-alpha*alpha)*(masses)/(beta))

    noise = np.random.normal(loc = 0, scale = sigv) 
    

    #define initial state 
    q_traj = np.zeros(n_steps // stride)
    v_traj = np.zeros(n_steps // stride)
    forces_traj = np.zeros(n_steps // stride)

        
    #collision rate is pjump/delt
    pjump = 0.025
    ncoll = 0
    cut = np.exp(-pjump)

    for step in tqdm(range(n_steps), desc = 'simulation timestep'):
        
        expdist = np.random.random(n_steps) > cut

        v = v * alpha + forces * (1-alpha) * friction + noise_scale * noise  
        
        q = q + v * dt 
    
        forces = du_dx(q, spl_m, values, q_init, width)

            
        #q, v, forces, potential = self.integrator.make_a_step(q, v, forces)
            
        # Test for a collision occurance
        if expdist[step]:
            ncoll = ncoll+1
            v = np.random.normal(loc=0,scale=sigv)

        if step % stride == 0:
            q_traj[(step - 1) // stride] = q
            v_traj[(step - 1) // stride] = v
            forces_traj[(step - 1) // stride] = forces

    return q_traj, v_traj, forces_traj