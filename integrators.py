import numpy as np
import sys
from utils import Forces_Potential
from forces import forces_potential_object
from typing import Dict, Optional, Callable
from numba import njit
import time 

class Integrator():
    """
    Class performs integration
    """
    
    def __init__():
        pass
    
    def make_a_step(self, q_old, v_old, forces_old):
        pass


class VelocityVerletIntegrator(Integrator):
    """Velocitiy Verlet Integrator"""

    def __init__(self, dt: float, friction: float, beta: float, masses: int, forcefield_name: str):
        self.dt = dt
        self.friction = friction
        self.beta = beta
        self.masses = masses
        self.forcefield_name = forcefield_name

    def make_a_step(self, q_old, v_old, forces_old):
        
        q_new = q_old + v_old * self.dt + 0.5 * forces_old * self.dt**2
        
        #forces, potential = Forces_Potential.forces_from_fes_surface(self.path, q_new)
        forces, potential = forces_potential_object(self.forcefield_name, q_new) 
        
        v_new = v_old + (forces + forces_old)/2 * self.dt
     
        return q_new, v_new, forces, potential 
    
    
class LangevinIntegrator(Integrator):
    """Langevin Integrator"""

    def __init__(self, dt: float, friction: float, beta: float, masses: int, forcefield_name: str):
        self.dt = dt
        self.friction = friction
        self.beta = beta
        self.masses = masses
        self.alpha = np.exp(-1*self.friction * self.dt) 
        self.noise_scale = np.sqrt(self.beta*(1-self.alpha**2)/self.masses)
        self.forcefield_name = forcefield_name

    def make_a_step(self, q_old, v_old, forces_old):

        noise = np.random.normal(loc = 0, scale=1) 

        v_new = v_old * self.alpha + forces_old * (1-self.alpha)*self.friction + self.noise_scale * noise  
        
        q_new = q_old + v_new * self.dt 
    
        forces, potential = forces_potential_object(self.forcefield_name, q_new) 

        return q_new, v_new, forces, potential 

class LangevinIntegratorBAOAB(Integrator):
    """Langevin Integrator with BAOAB scheme https://doi.org/10.1007/978-3-319-16375-8_ is used, where::

        B = deterministic velocity update
        A = deterministic position update
        O = stochastic velocity update
    """
    
    def __init__(self, dt: float,  friction: float, beta: float, masses: int, forcefield_name: str):
        self.dt = dt
        self.friction = friction
        self.beta = beta
        self.masses = masses
        self.alpha = np.exp(-self.friction * self.dt)
        self.noise_scale = np.sqrt(self.beta*(1-self.alpha**2)/self.masses)
        self.forcefield_name = forcefield_name
        
    
    def make_a_step(self, q_old, v_old, forces_old):
        
        #B
        v_new = v_old + 0.5*self.dt/self.masses * forces_old 
        #A
        q_new = q_old + 0.5*self.dt*v_new
        #O
        noise = np.random.normal(loc = 0, scale=1) 
        
        v_new = self.alpha * v_new + self.noise_scale * noise 
        #A
        q_new = q_new + 0.5 * self.dt * v_new
        #B
        
        forces, potential = forces_potential_object(self.forcefield_name, q_new)
        
        v_new = v_new + 0.5 * self.dt/self.masses * forces 


        return q_new, v_new, forces, potential