import numpy as np
from typing import Optional
from scipy.interpolate import interp1d, CubicSpline

class Forces_Potential():
    def V_chandler(q, params = dict):
        """Chandler potential energy function
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
            return wA**2*(q+qA)**2/2.0
        if (q > b):
            return VB+wB**2*(q-qB)**2/2.0
        else:
            return Q - q**2/2.0
    
    def F_chandler(q, params = dict):
        """Chandler potential force function with same parameters as for potential function
            Fq is scaled by 1/w1**2 so F/m is dimensionless
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
            return -wA**2*(q+qA)
        if (q > b):
            return -wB**2*(q-qB)
        else:
            return q

    def qA_qB(Q,VB,wA,wB):
        """helper function for qA and qB"""
        qA = np.sqrt(2*Q*(wA**2+1)/wA**2)
        qB = np.sqrt(2*(Q-VB)*(wB**2+1)/wB**2)
        return qA, qB
        
    def V_harmonic(q,params = dict):
        """" Harmonic Potential Function """
        k = params["k"]

        return 0.5*k*q**2

    def F_harmonic(q,params = dict):
        """" Harmonic Potential Function """
        k = params["k"]
        
        return k*q
        

    def calculate_potential_and_forces(forcefield, pot, q, params):
        """" Wrapper function for different potetnials and forces """
        
        forces = forcefield(q, params)
        potential = pot(q, params)

        return forces, potential
    
    
    def forces_from_fes_surface(path, q):
        fes = np.genfromtxt(path)

        rc = fes[:,0]
        pot = fes[:,1]

        cs = CubicSpline(rc, pot)

        potential_on_q  = cs(q)
        forces_on_q = cs(q, 1)

        return forces_on_q, potential_on_q 


    def calculate_kinetic_energy(velocity_trajectory, m=1):
        """
        Calculate kinetic energy as a function of time. 
        Parametes: 
        velocity_trajectory : iterable
            velocity_trajectory is an iterable where each element is a 2d np.ndarray,
            with nxm dimentions, where n is number of particles and m is dimentionality of the problem
        """
        return np.array([ 0.5*m*np.sum(np.square(i)) for i in velocity_trajectory])


    def set_velocities_to_temperature(T):
        """
        Initialize velocities according to Maxwel distribution
        Components are taken from normal distribution with mean 0 and 
        standard deviation 
        """
        v = np.random.normal(loc=0, scale=np.sqrt(1.0*T))
        print(v)
        v_mean = np.mean(v, axis=0)
        v = v - v_mean
        assert np.all(np.isclose(np.mean(v, axis=0), 0.0))
        return v
    
    def prinz_potential_V(q):
        """"V(x) = 4 ( x^8 + 0.8 e^{-80 x^2} + 0.2 e^{-80 (x-0.5)^2} + 0.5 e^{-40 (x+0.5)^2})"""
        potential = 4 * (q**8 + 0.8 * np.exp(-80 * q**2)+ 0.2 * np.exp(-80 *(q - 0.5)**2) + 0.5*np.exp(-40 *(q + 0.5)**2))
        forces = 32*q**7 - 512*q*np.exp(-80*q**2) + 0.8 * (80 - 160*q)*np.exp(-80 *(q - 0.5)**2) + 2 * (-80*q-40)*np.exp(-40*(q + 0.5)**2)
        
        return potential
    
    def prinz_potential_F(q):
        """"V(x) = 4 ( x^8 + 0.8 e^{-80 x^2} + 0.2 e^{-80 (x-0.5)^2} + 0.5 e^{-40 (x+0.5)^2})"""
        potential = 4 * (q**8 + 0.8 * np.exp(-80 * q**2)+ 0.2 * np.exp(-80 *(q - 0.5)**2) + 0.5*np.exp(-40 *(q + 0.5)**2))
        forces = 32*q**7 - 512*q*np.exp(-80*q**2) + 0.8 * (80 - 160*q)*np.exp(-80 *(q - 0.5)**2) + 2 * (-80*q-40)*np.exp(-40*(q + 0.5)**2)
        
        return forces
    

    def triple_well(q):
        """ V(x) = 5 - 24.82 x + 41.4251 x^2 - 27.5344 x^3 + 8.53128 x^4 - 1.24006 x^5 + 0.0684 x^6."""
        return 5 - 24.82 *q + 41.4251 *q**2 - 27.5344*q**3 + 8.53128*q**4 - 1.24006*q**5 + 0.0684*q**6

    def double_well(q,a):
        """(x_1^2 - 1)^2 + x_2^2."""
        return (q**2 - a**2)**2 
