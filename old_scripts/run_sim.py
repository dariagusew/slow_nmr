from jsonargparse import CLI
import sys
from integrators import Integrator, VelocityVerletIntegrator, LangevinIntegrator, LangevinIntegratorBAOAB
from simulation import Simulation
from utils import Forces_Potential
from typing import Dict, Optional, Callable, Union
from time import ctime
from types import NoneType
import numpy as np

def run_sim(integrator: Integrator, 
            n_steps: int,
            stride: int,  
            forcefield_name: str,
            q_init: Optional[float],
            v_init: Optional[float],
            last_ckpt: bool):

    
    int = integrator

    simulation = Simulation(int)
   
    q_traj, v_traj, forces = simulation.run_sim(n_steps, stride, forcefield_name, q_init, v_init, last_ckpt)



if __name__ == "__main__":
    print("Start run_sim.py: {}".format(ctime()))

    CLI(run_sim)

    print("Finish run_sim.py: {}".format(ctime()))