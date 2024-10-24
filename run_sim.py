from jsonargparse import CLI
import numpy as np
import scipy
from simulation import sim
from time import ctime


#run_sim(q_init=q_init,friction=1,masses=mass,beta=0.4,dt=dt,n_steps=n_steps,stride=stride,spl_m=spl_m,values=rc,width=width)
#run_sim(q_init=q_init,friction=0.5,masses=mass,beta=0.4,dt=dt,n_steps=n_steps,stride=stride,spl_m=spl_m)


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
    pot = fes[:,1]
    spl_m = scipy.interpolate.CubicSpline(rc, pot)

    
   
    q_traj, v_traj, forces = sim(q_init=q_init,
                                 friction=friction,
                                 masses=masses,
                                 beta=beta,
                                 dt=dt,
                                 n_steps=n_steps,
                                 stride=stride,
                                 spl_m=spl_m)



if __name__ == "__main__":
    print("Start run_sim.py: {}".format(ctime()))

    CLI(run_sim)

    print("Finish run_sim.py: {}".format(ctime()))