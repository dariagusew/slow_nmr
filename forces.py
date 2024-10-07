import numpy as np
from scipy.interpolate import CubicSpline


def F_V_chandler(q, params = dict):
    """Chandler force/potential energy function
    Vq = PE in units of kT
    q = generalized coordinate, dimensionless q = x sqrt(m w1^2/(kb T))
    m = mass
    Q = barrier height in units of kT
    VB = difference between minima in units of kT
    qA, qB locations of well minima in dimensionless units 
        
    Note the following restrictions:
    qA > sqrt(2Q)
    qB > sqrt(2(Q-VB))
        
    All variables are dimensionless per definition 
        
    """
    Q = params["Q"]
    VB = params["VB"]
    qA = params["qA"]
    qB = params["qB"]
        
    a = 2*Q/qA

    b = 2*(Q - VB)/qB

    wA = np.sqrt(a/(qA - a))
    wB = np.sqrt(b/(qB - b))

    if (q < -a):
        return -wA**2*(q+qA), wA**2*(q+qA)**2/2.0
    if (q > b):
        return -wB**2*(q-qB), VB+wB**2*(q-qB)**2/2.0
    else:
        return q, Q - q**2/2.0
    
# def F_chandler(q, params = dict):
#     """Chandler potential force function with same parameters as for potential function
#         Fq is scaled by 1/w1**2 so F/m is dimensionless
#     """
#     Q = params["Q"]
#     VB = params["VB"]
#     qA = params["qA"]
#     qB = params["qB"]
        
#     a = 2*Q/qA

#     b = 2*(Q - VB)/qB

#     wA = np.sqrt(a/(qA - a))
#     wB = np.sqrt(b/(qB - b))

#     if (q < -a):
#         return -wA**2*(q+qA)
#     if (q > b):
#         return -wB**2*(q-qB)
#     else:
#         return q
    

def forces_from_fes_surface(q, path):
    
    fes = np.genfromtxt(path)

    rc = fes[:,0]
    pot = fes[:,1]

    cs = CubicSpline(rc, pot)

    potential_on_q  = cs(q)
    forces_on_q = -cs(q, 1)

    return forces_on_q, potential_on_q 


def forces_potential_object(forcefield_name, q):
     
    if forcefield_name == 'chandler':
        params = dict({"Q":7,"VB":2,"qA":8.4,"qB":13.04})
        return F_V_chandler(q, params)
    
    if forcefield_name == 'fes':
        path = '/Users/zvh378/Desktop/Projects/2024/slow_nmr/forcefields/fes_667.dat' 
        return forces_from_fes_surface(q, path)