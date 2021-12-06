import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import seaborn as sns


def SEIRVAC(beta,beta_v, p, p_v):
    # Parametros
    T_inc = 9  # Periodo de incubación(dias)
    delta = 1 / T_inc  # Tasa a la cual una persona deja la clase de expuestos. delta tasa de transferencia E->I
    T_inf = 3  # Periodo de infección (dias)
    gamma = 1 / T_inf  # Tasa de recuperación. gamma es la tasa de transferencia de→I->R

    alpha = 1 / 100  # porción que mueren por dia (de cada 25 personas infectados uno muere por dia)
    alpha_v = 5 / 100  # porción que mueren por dia (de cada 25 personas infectados uno muere por dia)

    rho = 1 / 12  # Tasa a la cual mueren debido a la enfermedad = 1/(tiempo desde que se infectó hasta la muerte)

    # El Modelo SEIRD
    # Para dos poblaciones tomamos como referencia:
    # https://www.math.u-bordeaux.fr/~pmagal100p/papers/MSW-SIAM-2016.pdf
    def SEIRD(X, t):  # X vector de variables de estado
        Su = X[0] # Numero de personas Susceptibles no vacunadas en el dia t
        E = X[1]  # Numero de personas expuestas en el dia t (supuesto son infecciosos)
        I = X[2]  # Numero de personas infectadas en el dia t
        R = X[3]  # Numero de personas recuperadas en el dia t
        D = X[4]  # Numero de personas muertas en el dia t
        # Para mayores
        Seu = X[5]  # Numero de personas Susceptibles no vacunadas en el dia t
        Ee = X[6]  # Numero de personas expuestas en el dia t (supuesto son infecciosos)
        Ie = X[7]  # Numero de personas infectadas en el dia t
        Re = X[8]  # Numero de personas recuperadas en el dia t
        De = X[9]  # Numero de personas muertas en el dia t
        dSudt = - beta * (I + Ie) * Su - beta * E * Su  # EDO que modela el cambio de S
        dEdt = beta * (I + Ie) * Su + beta * E * Su - delta * E  # EDO que modela el cambio de E
        dIdt = delta * E - gamma * (1 - alpha) * I - rho * alpha * I  # EDO que modela el cambio de I
        dRdt = gamma * (1 - alpha) * I  # EDO que modela el cambio de R
        dDdt = rho * alpha * I  # EDO que modela el cambio de D
        #
        dSeudt = - beta_v * (I + Ie) * Seu - beta_v * Ee * Seu  # EDO que modela el cambio de S
        dEedt = beta_v * (I + Ie) * Seu + beta_v * Ee * Seu - delta * Ee  # EDO que modela el cambio de E
        dIedt = delta * Ee - gamma * (1 - alpha_v) * Ie - rho * alpha_v * Ie  # EDO que modela el cambio de I
        dRedt = gamma * (1 - alpha_v) * Ie  # EDO que modela el cambio de R
        dDedt = rho * alpha_v * Ie  # EDO que modela el cambio de D

        z = [dSudt, dEdt, dIdt, dRdt, dDdt,
             dSeudt, dEedt, dIedt, dRedt, dDedt]
        return z

    # Condiciones iniciales y Simulación
    S0 = 0.99
    Sv0 = S0 * p
    Su0 = S0 * (1 - p)
    E0 = 0.01
    I0 = 0
    R0 = 0
    D0 = 0
    Se0 = 0.99
    Sev0 = Se0 * p_v
    Seu0 = S0 * (1 - p_v)
    Ee0 = 0.01
    Ie0 = 0
    Re0 = 0
    De0 = 0

    ICS = [Su0, E0, I0, R0, D0,
           Seu0, Ee0, Ie0, Re0, De0]
    tF = 90  # 3 meses
    t = np.linspace(0, tF, tF)  # OJO!. Observaciones diarias
    # Solucion numerica del sistema de EDO y extarccion de las soluciones
    SOL = odeint(SEIRD, ICS, t)
    Su = SOL[:, 0]
    E = SOL[:, 1]
    I = SOL[:, 2]
    R = SOL[:, 3]
    D = SOL[:, 4]
    Seu = SOL[:, 5]
    Ee = SOL[:, 6]
    Ie = SOL[:, 7]
    Re = SOL[:, 8]
    De = SOL[:, 9]
    md = max(D) * 100  # Este número es arbitrario, solo para representar una población joven de 100 personas
    mde= max(De) * 100 # Este número es arbitrario, solo para representar una población mayor de 100 personas
    return md+mde

##
plt.figure()
x = np.linspace(0,1,50)
y = np.linspace(0,1,50)
N = np.size(x,0)
z = np.zeros([N,N])
Beta_list_j = [0.1,2,4,8]
# Plot options
num_ticks = 5
# the index of the position of yticks
yticks = np.linspace(0, len(x) - 1, num_ticks, dtype= int)
# the content of labels of these yticks
yticklabels = [x[idx] for idx in yticks]
yticklabels = np.round(yticklabels,2)
k = 1
for Beta in Beta_list_j:
    for i in range(N):
        for j in range(N):
           z[i,j] = SEIRVAC(Beta,0.1,x[i],y[j])
    plt.subplot(2,2,k)
    k += 1
    ax = sns.heatmap(z, xticklabels=yticklabels, yticklabels=yticklabels,vmin=0, vmax=1)
    ax.set_yticks(yticks)
    ax.set_xticks(yticks)
    ax.set(yticklabels=yticklabels,
           xticklabels=yticklabels,
           xlabel='Proporción jóvenes vacunados',
           ylabel='Proporción Mayores vacunados',
           title = 'Beta_j= {price:.1f} vs Beta_v=0.1'.format(price=Beta))
    ax.invert_yaxis()

##

