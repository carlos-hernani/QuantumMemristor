
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
import qutip as qp
import time
import pickle

from QM_evol import memristor_evolution as me



# We parametrize inital state as = cos(theta_1/2)|0> + e^(i*theta_2)sin(theta_1/2)|1>
theta_1 = np.pi/2  # Fix theta_1 to maximize initial voltage

# Range of values for theta_2 and alpha
theta2_min = 0.01
theta2_max = 0.99*2*np.pi

alpha_min = 0.1
alpha_max = 100


# Preallocation
n_samples = 2000  # Number of random samples to consider

# Store the relevant data into a structured array
Data_dtype = np.dtype([('theta_2', np.float32), 
                       ('alpha', np.float32),
                       ('Formfactor', np.float32)])

Data = np.zeros(n_samples, dtype = Data_dtype)

V = []
Iqp = []

for k1 in range(n_samples):
    theta_2 = np.random.rand()*(theta2_max - theta2_min) + theta2_min  # random theta_2 value
    alpha = np.random.rand()*(alpha_max - alpha_min) + alpha_min  # random alpha value
    
    FF, V_aux, Iqp_aux, t, indices, parameters = me.Single_memristor_evolution(theta_1 = theta_1, theta_2 = theta_2, alpha = alpha)
    Data['theta_2'][k1] = theta_2
    Data['alpha'][k1] = alpha
    Data['Formfactor'][k1] = FF
    V.append(V_aux)
    Iqp.append(Iqp_aux)

    
omega = parameters['omega']
I0 = parameters['I0']
V0 = parameters['V0']
timescale = parameters['timescale']


with open('data/Single_memristor_formfactor.dat', 'wb') as arg:
    pickle.dump([Data, V, Iqp, time, parameters], arg)

    
    
