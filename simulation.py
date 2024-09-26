import numpy as np 
import sys
from utils import Forces_Potential
from typing import Dict, Optional, Callable
from typing import Union, Optional
from numba import njit
from forces import forces_potential_object
import time

class Simulation():

    def __init__(self, integrator):
        self.integrator = integrator
        
    def run_sim(self, n_steps, stride, q_init, forcefield_name, v_init = Optional[float], sigv = 1, ncoll = 0):
        
        #define initial state 
        q_traj = []
        v_traj = []
        #u_traj = []


        #define initial positions and velocities 
        q = q_init

        v_init = np.random.normal(loc=0,scale=sigv)
        v = v_init

        start = time.time()
        forces, potential = forces_potential_object(forcefield_name, q)
        end = time.time()

        print("Elapsed force calc. = %s" % (end - start))

        #collision rate is pjump/delt
        pjump = 0.025
        cut = np.exp(-pjump)
        
        start = time.time()

        for i in range(n_steps):
            if (i%(n_steps/10) == 0):
                print('iteration ',i/(n_steps/10))
            
            expdist = np.random.random(stride) > cut
            for k in range(stride):
                q, v, forces, potential = self.integrator.make_a_step(q, v, forces)
            
                # Test for a collision occurance
                if expdist[k]:
                    ncoll = ncoll+1
                    v = np.random.normal(loc=0,scale=sigv)
            
            q_traj.append(q)
            v_traj.append(v)

        end = time.time() 
        print("Elapsed Langevin loop = %s" % (end - start))   

        return q_traj, v_traj