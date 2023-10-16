import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import qutip as qp

from QM_evol import memristor_evolution as me

# Load data
Data_best = pd.read_csv(r'data\best_ff_75percentile.csv')
Data_worst = pd.read_csv(r'data\worst_ff_25percentile.csv')

# single quantum memristor
Data_sqm = Data_best.loc[(Data_best['C12'] == 0) & (Data_best['L12'] == 0)]
idx_sqm = Data_sqm['Formfactor_1'].idxmax()  # index of the row with highest form factor for single quantum memristor
parameters_sqm = [[np.pi/2, Data_sqm['theta'].iloc[idx_sqm]],  # theta_1
                   [np.pi/2, Data_sqm['theta'].iloc[idx_sqm]],  # theta_2
                   Data_sqm['lambda_'].iloc[idx_sqm]]  # lambda

# Data best case
idx_best = Data_best['Formfactor_1'].idxmax()
parameters_best = [[np.pi/2, Data_best['theta'].iloc[idx_best]],  # theta_1
                   [np.pi/2, Data_best['theta'].iloc[idx_best]],  # theta_2
                   Data_best['lambda_'].iloc[idx_best],  # lambda
                   Data_best['C12'].iloc[idx_best],  # C12
                   Data_best['L12'].iloc[idx_best]]  # L12

# Data worst case
idx_worst = Data_worst['Formfactor_1'].idxmin()
parameters_worst = [[np.pi/2, Data_best['theta'].iloc[idx_worst]],  # theta_1
                   [np.pi/2, Data_best['theta'].iloc[idx_worst]],  # theta_2
                   Data_best['lambda_'].iloc[idx_worst],  # lambda
                   Data_best['C12'].iloc[idx_worst],  # C12
                   Data_best['L12'].iloc[idx_worst]]  # L12


#  Calculate the evolution with the best parameters
Rho_sqm, Results_sqm, Params_sqm = me.Coupled_memristor_evolution(*parameters_sqm, n_osci=10) # evolves two identical and uncoupled quantum memristors
# equivalent to evolving a single quantum memristor

# Set form factor, voltage and current into different variables
FF_sqm, V_aux_sqm, Iqp_aux_sqm = Results_sqm
t_sqm, indices_sqm, params_sqm = Params_sqm

# Put the best parameters into separate variables
omega1_sqm = params_sqm['omega1']
omega2_sqm = params_sqm['omega2']
I1_0_sqm = params_sqm['I1_0']
I2_0_sqm = params_sqm['I2_0']
V1_0_sqm = params_sqm['V1_0']
V2_0_sqm = params_sqm['V2_0']
timescale_sqm = params_sqm['timescale']


# Calculate evolution for coupled quantum memristors
n_oscillations = 20
Rho_best, Results_best, Params_best = me.Coupled_memristor_evolution(*parameters_best, n_osci=n_oscillations) 
Rho_worst, Results_worst, Params_worst = me.Coupled_memristor_evolution(*parameters_worst, n_osci=n_oscillations) 

# Best
FF_best, V_aux_best, Iqp_aux_best = Results_best
t_best, indices_best, params_best = Params_best

omega1_best = params_best['omega1']
omega2_best = params_best['omega2']
I1_0_best = params_best['I1_0']
I2_0_best = params_best['I2_0']
V1_0_best = params_best['V1_0']
V2_0_best = params_best['V2_0']
timescale_best = params_best['timescale']

# Worst
FF_worst, V_aux_worst, Iqp_aux_worst = Results_worst
t_worst, indices_worst, params_worst = Params_worst

omega1_worst = params_worst['omega1']
omega2_worst = params_worst['omega2']
I1_0_worst = params_worst['I1_0']
I2_0_worst = params_worst['I2_0']
V1_0_worst = params_worst['V1_0']
V2_0_worst = params_worst['V2_0']
timescale_worst = params_worst['timescale']


# Calculate concurrence
FF_1_best = FF_best[0]
FF_2_best = FF_best[1]

FF_1_worst = FF_worst[0]
FF_2_worst = FF_worst[1]


concurrence_best = np.zeros(len(Rho_best.states))
concurrence_worst = np.zeros(len(Rho_worst.states))

for t1, state in enumerate(Rho_best.states):
    concurrence_best[t1] = qp.concurrence(state)
    
for t1, state in enumerate(Rho_worst.states):
    concurrence_worst[t1] = qp.concurrence(state)


# Generate figures
# Fig 1
fig1 = plt.figure(1)

ax_sqm1 = fig1.add_subplot(1,2,1)
ax_sqm2 = fig1.add_subplot(1,2,2)

count = 10*500
ax_sqm1.plot(V_aux_sqm[0][0:count]/V1_0_sqm, Iqp_aux_sqm[0][0:count]/I1_0_sqm,
         lw = 1.5, color='black')
ax_sqm1.set_xlabel(r'$V/V_0$', fontsize = 20, labelpad = 10)
ax_sqm1.set_ylabel(r'$I/I_0$', fontsize = 20, labelpad = 10)
ax_sqm1.tick_params(axis='both', which='major', labelsize=15, width = 2, length=4)

ax_sqm2.plot(t_sqm[indices_sqm[1:]]/timescale_sqm, FF_sqm[0], lw=1, ls='--', marker='o')
ax_sqm2.set_ylim([0, 0.35])
ax_sqm2.set_xlabel(r'$t/T$', fontsize = 25, labelpad = 10)
ax_sqm2.set_ylabel(r'$\mathcal{F}$', fontsize = 25, labelpad = 10)
ax_sqm2.tick_params(axis='both', which='major', labelsize=20, width = 2, length=4)

plt.tight_layout()


# ---- Fig2 ----
fig3 = plt.figure(3)

ax31 = fig3.add_subplot(1,2,1)
ax32 = fig3.add_subplot(1,2,2)

# best case 
count = 10*500
ax31.plot(V_aux_best[0][0:count]/V1_0_best, Iqp_aux_best[0][0:count]/I1_0_best,
         lw = 1.5, label='optimal', color='black')
ax31.set_title('optimal', fontsize = 20)
ax31.set_xlabel(r'$V/V_0$', fontsize = 20, labelpad = 10)
ax31.set_ylabel(r'$I/I_0$', fontsize = 20, labelpad = 10)
ax31.tick_params(axis='both', which='major', labelsize=15, width = 2, length=4)


# worst case 
ax32.plot(V_aux_worst[1][0:count]/V2_0_worst, Iqp_aux_worst[1][0:count]/I2_0_worst,
        lw = 1.5, label='suboptimal', color='black')
ax32.set_title('suboptimal', fontsize = 20)
ax32.set_xlabel(r'$V/V_0$', fontsize = 20, labelpad = 10)
ax32.set_ylabel(r'$I/I_0$', fontsize = 20, labelpad = 10)
ax32.tick_params(axis='both', which='major', labelsize=15, width = 2, length=4)
plt.tight_layout()


# ---- Fig 4 --- 
fig4 = plt.figure(4)

ax41 = fig4.add_subplot(1,1,1)

ax41.plot(t_best[indices_best[1:]]/timescale_best, FF_best[0], lw=1, marker='o', label='optimal', color='black')
ax41.plot(t_worst[indices_worst[1:]]/timescale_worst, FF_worst[0], lw=1, ls='--', marker='o', label='suboptimal',color='grey')
# ax41.set_title('Form factor', fontsize = 25)
ax41.set_xlabel(r'$t/T$', fontsize = 25, labelpad = 10)
ax41.set_ylabel(r'$\mathcal{F}$', fontsize = 25, labelpad = 10)
ax41.tick_params(axis='both', which='major', labelsize=20, width = 2, length=4)
plt.legend(fontsize=20)


# ----  Fig 5  ----- 
fig5 = plt.figure(5, figsize = [12, 8])
ax5 = fig5.add_subplot(1,1,1)
ax5.plot(t_best/timescale_best, concurrence_best, lw = 3, label='optimal', color='black')
ax5.plot(t_worst/timescale_worst, concurrence_worst, lw = 3, label='suboptimal', ls='--', color='grey')
ax5.set_xlabel(r'$t/T$', fontsize = 20, labelpad = 10)
ax5.set_ylabel(r'$C$', fontsize = 20, labelpad = 10)
ax5.tick_params(axis='both', which='major', labelsize=15, width = 2, length=4)
ax5.legend(fontsize=20)