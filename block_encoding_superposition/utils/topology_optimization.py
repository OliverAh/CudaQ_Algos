import numpy as np


def construct_local_stiffness():
    E=1
    nu=0.3 # Poisson's ratio (standard = 0.3)
    k=np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
    local_stiffness = E/(1-nu**2)*np.array([
        [k[0], k[5], k[6], k[3], k[2], k[7], k[4], k[1]],
        [k[5], k[0], k[7], k[2], k[3], k[6], k[1], k[4]],
        [k[6], k[7], k[0], k[1], k[4], k[5], k[2], k[3]],
        [k[3], k[2], k[1], k[0], k[5], k[4], k[7], k[6]],
        [k[2], k[3], k[4], k[5], k[0], k[1], k[6], k[7]],
        [k[7], k[6], k[5], k[4], k[1], k[0], k[3], k[2]],
        [k[4], k[1], k[2], k[7], k[6], k[3], k[0], k[5]],
        [k[1], k[4], k[3], k[6], k[7], k[2], k[5], k[0]] ])
    return local_stiffness
