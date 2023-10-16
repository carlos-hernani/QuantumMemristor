
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
import qutip as qp
import pickle

from QM_evol import memristor_evolution as me

# System Parameters 
# Range of values for theta, alpha, C12 and L12
theta_min = 0.01
theta_max = 0.99*2*np.pi

lambda_min = 0.1
lambda_max = 100

C12_min = 0           # minimum value for capacitive coupling
C12_max = 20 * 1e-13  # maximum value for capacitive coupling

L12_min = 0           # minimum value for inductive coupling
L12_max = 20 * 1e-9   # maximum value for inductive coupling


# Preallocation
n_samples = 5000
Data_dtype = np.dtype([('C12', np.float32),
                       ('L12', np.float32),
                       ('theta', np.float32),
                       ('lambda_', np.float32),
                       ('Formfactor_1', np.float32),
                       ('Formfactor_2', np.float32)])

Data = np.zeros(n_samples, dtype = Data_dtype)

Time = []
V = []
Iqp = []

omega1 = np.zeros(n_samples)
omega2 = np.zeros(n_samples)
I1_0 = np.zeros(n_samples)
I2_0 = np.zeros(n_samples)
V1_0 = np.zeros(n_samples)
V2_0 = np.zeros(n_samples)
timescale = np.zeros(n_samples)

for k1 in range(n_samples):
    C12 = np.random.uniform(low = C12_min, high = C12_max)
    L12 = np.random.uniform(low = L12_min, high = L12_max)
    theta = np.random.uniform(low = theta_min, high = theta_max)
    lambda_ = np.random.uniform(low = lambda_min, high = lambda_max)
    
    theta1 = [np.pi/2, theta]
    theta2 = [np.pi/2, theta]
    
    Results, Parameters = me.Coupled_memristor_evolution(theta_1 = theta1, theta_2 = theta2, lambda_ = lambda_,
                                                         C12 = C12, L12 = L12)
    
    FF, V_aux, Iqp_aux = Results
    t, indices, parameters = Parameters
    Data['C12'][k1] = C12
    Data['L12'][k1] = L12
    Data['theta'][k1] = theta
    Data['lambda_'][k1] = lambda_
    Data['Formfactor_1'][k1] = FF[0]
    Data['Formfactor_2'][k1] = FF[1]
    V.append(V_aux)
    Iqp.append(Iqp_aux)
    Time.append

    omega1[k1] = parameters['omega1']
    omega2[k1] = parameters['omega2']
    I1_0[k1] = parameters['I1_0']
    I2_0[k1] = parameters['I2_0']
    V1_0[k1] = parameters['V1_0']
    V2_0[k1] = parameters['V2_0']
    timescale[k1] = parameters['timescale']
# end of k1 loop



with open('Coupled_memristors_formfactor.dat', 'wb') as arg:
    pickle.dump([Data, V, Iqp, Time, parameters], arg)
    
    