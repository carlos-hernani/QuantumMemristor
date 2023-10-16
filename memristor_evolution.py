
import numpy as np
import qutip as qp


def Single_memristor_evolution(theta_1, theta_2, alpha):
    # =============================================================================
    # System Parameters
    # =============================================================================
    """ Fundamental constants"""
    hbar = 1.054571 * 1e-34  # Planck constant [J x s]
    phi0 = 2.067833 * 1e-15/(2*np.pi)  # Magnetic Flux quantum [Wb]
    G0 = 7.748091 * 1e-5  # Conductance quantum [Siemens]
    e = 1.602176 * 1e-19 # electron charge [C]
    
    "Circuit Parameters"
    # Based on the reference, the Capacitive energy is 1Ghz, and the
    # Inductive energy is 1000 times the capacitive energy. 
    # We choose the parameters corresponding to this condition
    Cg = 4.868269*1e-13  # Effective SQUID capacitance [Farad] => Cg = 2*(e**2)/(1e9*hbar)
    L = 1.027059*1e-9  # Inductance of outer loop [Henry]
    EC = 2 * (e**2)/Cg  # Charging energy in SQUID capacitor
    EL = (phi0**2)/L    # Inductive energy
    g0 = (EC/(32*EL))**(1/4) # Constant relevant for n and phi operators
    omega = np.sqrt(2*EC*EL)/(1e9*hbar) # Frequency of the harmonic oscillator, in units of GHz
    s_w = alpha * omega  # Spectral density  of environment
    nmax = 1  # Number of excitations in the harmonic oscillator
    
    # =============================================================================
    # Operators and decay
    # =============================================================================
    """ Define the operators and Hamiltonian of the system """
    a = qp.destroy(nmax+1)  # Annihilation operator
    I = qp.qeye(nmax+1)   # Identity
    H = omega*a.dag()*a   # Hamiltonian
 
    n = (1j/(4*g0))*(a.dag() - a) # Number of Cooper pairs operator
    phi = 2*g0*(a.dag() + a)   # Phase operator

    "Initial State"
    psi_0 = np.cos(theta_1/2)*qp.fock(nmax+1,1) + np.exp(1j*theta_2)*np.sin(theta_1/2) * qp.fock(nmax+1,0) 


    """ Define the time dependent decay rate """
    params = {'g0':g0}

    def gamma_t(t,args):
        g0 = args['g0']  # import g0 into function namespace
        phi_ini = np.pi/2  # phase shift of external flux
        phid = phi_ini + np.sin(omega*t)  # external flux
        gamma_0 = 0.25*g0**2 * np.exp(-g0**2) * s_w   # bare decay rate
        gamma_T = np.sqrt(gamma_0 * (1 + np.cos(phid)))  # We take square root because master equation solver squares the decay rate

        return gamma_T

    def gamma(t, args):
        g0 = args['g0']  # import g0 into function namespace
        phi_ini = np.pi/2  # phase shift of external flux
        phid = phi_ini + np.sin(omega*t)  # external flux
        gamma_0 = 0.25*g0**2 * np.exp(-g0**2) * s_w   # bare decay rate
        gamma = gamma_0 * (1 + np.cos(phid))  # We take square root because master equation solver squares the decay rate

        return gamma

    "Collapse operators"
    c_ops = [[a, gamma_t]]  # time-dependent collapse term

    # =============================================================================
    # System Dynamics
    # =============================================================================
    """ Dynamics parameters """
    number_oscillations = 5
    points_per_oscillation = 500
    time_points = number_oscillations * points_per_oscillation
    timescale = 2*np.pi/(omega)  # Period of the oscillator
    t = timescale * np.linspace(0, number_oscillations, time_points) # Time vector for our time evolution

    rho_t = qp.mesolve(H, psi_0, t, c_ops, args=params)  # Solve master equation

    V = np.zeros(len(t))  # Voltage vector
    P = np.zeros(len(t))  # Phase vector
    Iqp = np.zeros(len(t))  # Quasiparticle current vector

    for i in range(len(t)):
        rho_ti = rho_t.states[i]
        V[i] = -2*e*qp.expect(n,rho_ti)/Cg    # Voltage
        P[i] = qp.expect(phi,rho_ti)   # Phase
        Iqp[i] = gamma(t[i], params)*V[i]*Cg   # Quasiparticle Current
        
    """ Define maximum initial voltage and quasiparticle current """
    psi_max_volt = np.cos(np.pi/4)*qp.fock(nmax+1,1) + np.exp(1j*np.pi/2)*np.sin(np.pi/4) * qp.fock(nmax+1,0) 
    
    V0 = -2*e*qp.expect(n,psi_max_volt)/Cg  # Normalizing voltage constant
    I0 = V0 * 0.25* g0**2 * np.exp(-g0**2)* (100*omega) * Cg     # Normalizing quasiparticle current constant considering alpha = 100
    
    # =============================================================================
    # Auxiliary functions
    # =============================================================================

    def Identify_loops(x,y):
        """
        This function stores the indices of the point in time
        when x crosses zero value, and stores them in the indices
        container.

        One memristive hysteresis curve crosses the origin two times during its entire loop
        So the indices that denote the starting point of each hysteresis curve are obtained
        by taking every other element from indices container.
        This is stored in the real_indices container.
        """
        indices = []
        real_indices = []
        for ind in range(len(x)-1):
            prod_sign = np.sign(x[ind]*x[ind+1])
            if prod_sign == -1:
                closest_ind = np.argmin(np.abs([x[ind], x[ind+1]])) # Choose which point is closer to zero n or n+1
                if closest_ind == 0:
                    indices.append(ind)
                else:
                    indices.append(ind+1)

        for n in range(int(len(indices)/2) ):
            real_indices.append(indices[2*n])  
        return real_indices  # 

    def Area(x,y):
        "Area is calculated with Green's theorem"
        A = 0
        for i in range(len(x)-1):
            A += 0.5*abs(y[i]*(x[i+1]-x[i]) - x[i]*(y[i+1]-y[i]))
        return A

    def Perimeter(x,y):
        L = 0
        for i in range(len(x)-1):
            L += np.sqrt((x[i+1] - x[i])**2 + (y[i+1] - y[i])**2)
        return L

    # =============================================================================
    # Form Factor
    # =============================================================================
    def Form_factor(x,y):
        indices = Identify_loops(x,y)
        number_of_loops = len(indices)-1
        form_factor = np.zeros([number_of_loops])

        for n in range(number_of_loops):
            loop_x = x[indices[n]:indices[n+1]]
            loop_y = y[indices[n]:indices[n+1]]
            form_factor[n] = 4*np.pi*Area(loop_x,loop_y)/(Perimeter(loop_x,loop_y)**2)

        return form_factor


    # =============================================================================
    # Calculate Form Factor for current Dynamics
    # =============================================================================

    FF = Form_factor(V/V0,Iqp/I0) # Form factor for normalized current and voltage

    indices = Identify_loops(V/V0,Iqp/I0)  # indices of points in time when a full loop is completed
    
    parameters = {'omega':omega, 'V0':V0, 'I0':I0, 'timescale':timescale} 
    
    return FF[0], V, Iqp, t, indices, parameters




def Coupled_memristor_evolution(theta_1 = 1, theta_2 = 1, lambda_ = 1, C12 = 0, L12 = 0):
    # Fundamental constants
    hbar = 1.054571 * 1e-34  # Planck constant [J x s]
    phi0 = 2.067833 * 1e-15/(2*np.pi)  # Magnetic Flux quantum [Wb]
    G0 = 7.748091 * 1e-5  # Conductance quantum [Siemens]
    e = 1.602176 * 1e-19 # electron charge [C]

    # Based on the reference, the Capacitive energy is 1Ghz, and the
    # Inductive energy is 1000 times the capacitive energy. 
    # We choose the parameters corresponding to this condition
    C1 = 4.868269*1e-13  # Effective SQUID capacitance [Farad] => Cg = 2*(e**2)/(1e9*hbar)
    C2 = 4.868269*1e-13  # Effective SQUID capacitance [Farad] => Cg = 2*(e**2)/(1e9*hbar)
    L1 = 1.027059*1e-9  # Inductance of outer loop [Henry]
    L2 = 1.027059*1e-9  # Inductance of outer loop [Henry]
    
    "Here we use C12, L12 arguments"
    if C12 != 0:
        # Capacitance matrix
        C = np.array([ [C1 + C12, -C12], 
                       [-C12, C2 + C12] ])
        EC12 = 2*(e**2)*C12       # charging energy of capacitive coupling
    else:
        C = np.array([ [C1 , 0], 
                       [0, C2] ])
        EC12 = 0
    # Inverse of capacitance matrix
    C_inv = np.linalg.inv(C)

    if L12 != 0:
        # inductance matrix
        L = np.array([ [1/L1 + 1/L12, -1/L12], 
                       [-1/L12, 1/L2 + 1/L12] ])
        EL12 = (phi0**2)/L12      # Inductive energy of inductive coupling
    else:
        L = np.array([ [1/L1, 0], 
                       [0, 1/L2] ])
        EL12 = 0


    EC1 = 2*(e**2)*C_inv[0,0]  # Charging energy of capacitor of memristor 1
    EC2 = 2*(e**2)*C_inv[1,1]   # Charging energy of capacitor of memristor 2

    EL1 = (phi0**2) * L[0,0]    # Inductive energy of inductor of memristor 1
    EL2 = (phi0**2) * L[1,1]    # Inductive energy of inductor of memristor 2
    

    g1 = (EC1/(32*EL1))**(1/4) # Constant relevant for n and phi operators
    g2 = (EC2/(32*EL2))**(1/4) # Constant relevant for n and phi operators
    omega1 = np.sqrt(2*EC1*EL1)/(1e9*hbar) # Frequency of the harmonic oscillator of memristor 1, in units of GHz
    omega2 = np.sqrt(2*EC2*EL2)/(1e9*hbar) # Frequency of the harmonic oscillator of memristor 2, in units of GHz
    
    "Here we use s_cte argument"
    s_w1 = lambda_ * omega1  # Spectral density  of environment
    s_w2 = lambda_ * omega2  # Spectral density  of environment

    alpha = EL12/np.sqrt(EL1*EL2)
    beta = EC12/np.sqrt(EC1*EC2)
    dim = 2  # Number of excitations in the harmonic oscillator

    phi0_1 = np.pi/2
    phi0_2 = np.pi/2
    A_1 = 1
    A_2 = 1

    k = np.sqrt(omega1*omega2)*(alpha + beta)

    # =============================================================================
    # Operators and decay
    # =============================================================================
    """ Define the operators and Hamiltonian of the system """
    I = qp.qeye(dim)     # Identity
    a1 = qp.tensor(qp.destroy(dim), I)  # Annihilation operator for memristor 1
    a2 = qp.tensor(I, qp.destroy(dim))  # Annihilation operator for memristor 2
    H = omega1*a1.dag()*a1 + omega2*a2.dag()*a2 + k*(a1.dag()*a2 + a1*a2.dag())  # Coupled memristors Hamiltonian
 
    n1 = (1j/(4*g1))*(a1.dag() - a1) # Number of Cooper pairs operator for memristor 1
    n2 = (1j/(4*g2))*(a2.dag() - a2) # Number of Cooper pairs operator for memristor 2
    phi1 = 2*g1*(a1.dag() + a1)   # Phase operator for memristor 1
    phi2 = 2*g2*(a2.dag() + a2)   # Phase operator for memristor 2

    "Initial State"
    "Here we use theta_1, theta_2 arguments"
    psi_1 = np.cos(theta_1[0]/2)*qp.fock(dim,0) + np.exp(1j*theta_1[1])*np.sin(theta_1[0]/2) * qp.fock(dim, 1) 
    psi_2 = np.cos(theta_2[0]/2)*qp.fock(dim,0) + np.exp(1j*theta_2[1])*np.sin(theta_2[0]/2) * qp.fock(dim, 1) 
    
    psi_0 = qp.tensor(psi_1, psi_2)

    """ Define the time dependent decay rate """
    params = {'g1':g1, 'g2':g2, 'omega1':omega1, 'omega2':omega2, 's_w1':s_w1, 's_w2':s_w2}

    def gamma1(t,args):
        g1 = args['g1']     # import g1 into function namespace
        s_w1 = args['s_w1'] # import s_w1 into function namespace   
        phi_ini = np.pi/2  # phase shift of external flux
        phid = phi_ini + np.sin(omega1*t)  # external flux
        gamma_0 = 0.25*g1**2 * np.exp(-g1**2) * s_w1   # bare decay rate
        gamma_T = np.sqrt(gamma_0 * (1 + np.cos(phid)))  # We take square root because master equation solver squares the decay rate

        return gamma_T
              
    def gamma2(t,args):
        g2 = args['g2']     # import g1 into function namespace
        s_w2 = args['s_w2'] # import s_w1 into function namespace   
        phi_ini = np.pi/2  # phase shift of external flux
        phid = phi_ini + np.sin(omega2*t)  # external flux
        gamma_0 = 0.25*g2**2 * np.exp(-g2**2) * s_w2   # bare decay rate
        gamma_T = np.sqrt(gamma_0 * (1 + np.cos(phid)))  # We take square root because master equation solver squares the decay rate

        return gamma_T
              
    "Collapse operators"
    c_ops = [ [a1, gamma1], [a2, gamma2] ]  # time-dependent collapse operators

    # =============================================================================
    # System Dynamics
    # =============================================================================
    """ Dynamics parameters """
    number_oscillations = 5
    points_per_oscillation = 500
    time_points = number_oscillations * points_per_oscillation
    "Timescale should be chosen carefully"          
    timescale = 2*np.pi/(omega1)  # Period of the oscillator
    t = timescale * np.linspace(0, number_oscillations, time_points) # Time vector for our time evolution

    rho_t = qp.mesolve(H, psi_0, t, c_ops, args=params)  # Solve master equation

    V1 = np.zeros(len(t))  # Voltage vector for mem 1
    V2 = np.zeros(len(t))  # Voltage vector for mem 2
    P1 = np.zeros(len(t))  # Phase vector for mem 1
    P2 = np.zeros(len(t))  # Phase vector for mem 2
    Iqp1 = np.zeros(len(t))  # Quasiparticle current vector for mem 1
    Iqp2 = np.zeros(len(t))  # Quasiparticle current vector for mem 2

    for index, state in enumerate(rho_t.states):
        V1[index] = -2*e*qp.expect(n1, state)/C1    # Voltage for mem 1
        V2[index] = -2*e*qp.expect(n2, state)/C2    # Voltage for mem 2
        P1[index] = qp.expect(phi1, state)   # Phase for mem 1
        P2[index] = qp.expect(phi2, state)   # Phase for mem 2  
        Iqp1[index] = (gamma1(t[index], params)**2) * (V1[index]*C1)   # Quasiparticle Current for mem 1
        Iqp2[index] = (gamma2(t[index], params)**2) * (V2[index]*C2)   # Quasiparticle Current for mem 2
        
    """ Define maximum initial voltage and quasiparticle current """
    psi1_max_volt = np.cos(np.pi/4)*qp.fock(dim, 0) + np.exp(1j*np.pi/2)*np.sin(np.pi/4) * qp.fock(dim, 1)               
    psi2_max_volt = np.cos(np.pi/4)*qp.fock(dim, 0) + np.exp(1j*np.pi/2)*np.sin(np.pi/4) * qp.fock(dim, 1) 
    psi_max = qp.tensor(psi1_max_volt, psi2_max_volt)
    
    V1_0 = -2*e*qp.expect(n1, psi_max)/C1  # Normalizing voltage constant for mem 1
    V2_0 = -2*e*qp.expect(n2, psi_max)/C2  # Normalizing voltage constant for mem 2          
    I1_0 = V1_0*(gamma1(0, params)**2)*C1     # Normalizing quasiparticle current constant for mem 1
    I2_0 = V2_0*(gamma2(0, params)**2)*C2     # Normalizing quasiparticle current constant for mem 2
    
    # =============================================================================
    # Auxiliary functions
    # =============================================================================

    def Identify_loops(x,y):
        """
        This function stores the indices of the point in time
        when x crosses zero value, and stores them in the indices
        container.

        One memristive hysteresis curve crosses the origin two times during its entire loop
        So the indices that denote the starting point of each hysteresis curve are obtained
        by taking every other element from indices container.
        This is stored in the real_indices container.
        """
        indices = []
        real_indices = []
        for ind in range(len(x)-1):
            prod_sign = np.sign(x[ind]*x[ind+1])
            if prod_sign == -1:
                closest_ind = np.argmin(np.abs([x[ind], x[ind+1]])) # Choose which point is closer to zero n or n+1
                if closest_ind == 0:
                    indices.append(ind)
                else:
                    indices.append(ind+1)

        for n in range(int(len(indices)/2) ):
            real_indices.append(indices[2*n])  
        return real_indices  # 

    def Area(x,y):
        "Area is calculated with Green's theorem"
        A = 0
        for i in range(len(x)-1):
            A += 0.5*abs(y[i]*(x[i+1]-x[i]) - x[i]*(y[i+1]-y[i]))
        return A

    def Perimeter(x,y):
        L = 0
        for i in range(len(x)-1):
            L += np.sqrt((x[i+1] - x[i])**2 + (y[i+1] - y[i])**2)
        return L

    # =============================================================================
    # Form Factor
    # =============================================================================
    def Form_factor(x,y):
        indices = Identify_loops(x,y)
        number_of_loops = len(indices)-1
        form_factor = np.zeros([number_of_loops])

        for n in range(number_of_loops):
            loop_x = x[indices[n]:indices[n+1]]
            loop_y = y[indices[n]:indices[n+1]]
            form_factor[n] = 4*np.pi*Area(loop_x,loop_y)/(Perimeter(loop_x,loop_y)**2)

        return form_factor


    # =============================================================================
    # Calculate Form Factor for current Dynamics
    # =============================================================================

    FF1 = Form_factor(V1/V1_0, Iqp1/I1_0) # Form factor for normalized current and voltage
    FF2 = Form_factor(V2/V2_0, Iqp2/I2_0) # Form factor for normalized current and voltage
    ff1 = np.mean(FF1)
    ff2 = np.mean(FF2)
    
    FF = [ff1, ff2]
    V = [V1, V2]
    Iqp = [Iqp1, Iqp2]
    
    indices = Identify_loops(V1/V1_0,Iqp1/I1_0)  # indices of points in time when a full loop is completed
    
    parameters = {'omega1':omega1, 'omega2':omega2, 'V1_0':V1_0, 'I1_0':I1_0, 
                  'V2_0':V2_0, 'I2_0':I2_0, 'timescale':timescale} 
    
    Results = [FF, V, Iqp]
    Parameters = [t, indices, parameters]
              
    return Results, Parameters