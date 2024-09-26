import numpy as np 
import pytest 

def prob(q, kb=1, T=1):
    hist, _ = np.histogram(q)
    
    F = -kb*T*np.log(hist)
    
    return F