import numpy as np 
import sys
from utils import Forces_Potential
from typing import Dict, Optional, Callable
from typing import Union, Optional
from forces import forces_potential_object
import time
from tqdm import tqdm

class Simulation():

    def __init__(self, integrator):
        self.integrator = integrator
        
    def run_sim(self, n_steps, stride, q_init, forcefield_name, v_init = Optional[float], sigv = 1, ncoll = 0):
        
        #define initial positions and velocities 
        q = q_init

        v_init = np.random.normal(loc=0,scale=sigv)
        v = v_init

        #define initial state 

        q_traj = [q_init]
        v_traj = [v_init]
        forces_traj = []

        start = time.time()
        forces, potential = forces_potential_object(forcefield_name, q)
        end = time.time()

        print("Elapsed force calc. = %s" % (end - start))

        #collision rate is pjump/delt
        pjump = 0.025
        ncoll = 0
        cut = np.exp(-pjump)
        
        start = time.time()

        for i in tqdm(range(n_steps), desc = 'simulation timestep'):
            #if (i%(n_steps/10) == 0):
             #   print('iteration ',i/(n_steps/10))
            
            expdist = np.random.random(n_steps) > cut
            
            q, v, forces, potential = self.integrator.make_a_step(q, v, forces)
            
            # Test for a collision occurance
            if expdist[i]:
                ncoll = ncoll+1
                v = np.random.normal(loc=0,scale=sigv)
            
            # save to arrays if relevant
            if i % stride == 0:
                q_traj.append(q)
                v_traj.append(v)
                forces_traj.append(forces)
                # save arrays
                np.save('q_traj', q_traj)
                np.save('v_traj', v_traj)
                np.save('f_traj', forces_traj)
        

        end = time.time() 
        print("Elapsed Langevin loop = %s" % (end - start))   

        return q_traj, v_traj, forces