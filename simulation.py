from Dynamics_SGLF import Dynamics_SGLF
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.integrate import solve_ivp

T_YEARS = 20.  #год
V_0 = 25. # а.е./год, скорость аппарата в инерц системе
DT = 0.1 #Шаг интегрирования 
t0 = 0.
z0 = 550. # расстояние от КА до Солнца  (FCS) , а.е.

# Орбитальные параметры экзопланеты
Orb_param_exo_array = np.array([
    1. ,  # большая полуось (а.е)
    1. ,  # период обращения (годы)
    0.0167086,  # эксцентриситет
    -240., 352.5, 37.5,  # аргументы орбиты (град)
    datetime(2021, 1, 2)  # начальное время (дата)
])

# Массы тел (в кг)
M_sun = 1.989 * 1e30  # масса Солнца (кг)
M_JSUN_array = np.array([
    1.8982e27,  # Юпитер
    5.6834e26,  # Сатурн
    8.6810e25,  # Уран
    1.02413e26  # Нептун
])

# Орбитальные параметры планет-гигантов
Orb_param_JSUN_array = np.array([
    [5.2044 , 11.862 , 0.0489, 100.464, 273.867, 1.303, datetime(2023, 1, 21)],
    [9.5826 , 29.4571 , 0.0565, 113.665, 339.392, 2.485, datetime(2032, 11, 29)],
    [19.2184 , 84.0205 , 0.046381, 74.006, 96.998857, 0.773, datetime(2050, 8, 19)],
    [30.07, 164.8 , 0.008678, 131.784, 276.336, 1.767957, datetime(2042, 9, 4)]
])

# Создаем объект Dynamics_SGLF
dynamics = Dynamics_SGLF(
    Orb_param_exo_array=Orb_param_exo_array,
    M_sun=M_sun,
    M_JSUN_array=M_JSUN_array,
    Orb_param_JSUN_array=Orb_param_JSUN_array,
    t0=t0,
    z0=z0
)

def Dynamics(dynamics, t):
    """
    Вычисляет матрицы преобразования и ускорение Солнца.
    """
    dr_sun_dt, d2r_sun_dt = dynamics.r_sun_deriv(t)
    S = dynamics.basis_FCS(t)[0]
    dS_dt, d2S_dt = dynamics.dS_dt(t)
    a_vec = dynamics.basis_FCS(t)[1]
    return S, dS_dt, d2S_dt, dr_sun_dt, d2r_sun_dt, a_vec

def dX_dt(t, X, dynamics):
    """
    Производные для интегрирования методом RK45.
    """
    p_FCS = X[:3]  # Позиция в SSB 
    dpdt_FCS = X[3:]  # Скорость в SSB
    S, dS_dt, d2S_dt, dr_sun_dt_SSB, d2r_sun_dt_SSB, a_vec = Dynamics(dynamics, t)
    D = (d2S_dt @ p_FCS) + 2 * (dS_dt @ dpdt_FCS)
    d2pdt2_FCS = - S.T @ d2r_sun_dt_SSB - S.T @ D
    return np.concatenate((dpdt_FCS, d2pdt2_FCS), axis=0)  


def simulate_motion(dynamics, T_YEARS, V_0, DT):
    """
    Интеграция уравнений движения методом RK45.
    """
    t0 = 0.
    S, dS_dt, d2S_dt, dRdt_SSB_0, d2Rd2t_SSB_0, a_vec = Dynamics(dynamics, t0)
    times = np.arange(0., T_YEARS, DT)
    # Начальные условия в системе FCS
    p0_FCS = np.array([0.0, 0.0, z0])  
    drdt_SSB_0 = V_0 * S[:, 2]
    dpdt_FCS_0 = S.T @ (drdt_SSB_0 - dRdt_SSB_0 - dS_dt @ p0_FCS)
    X0 = np.concatenate((p0_FCS, dpdt_FCS_0))
    print(dS_dt @ S.T + S @ dS_dt.T)
    #exit()
    # Запуск RK45
    sol = solve_ivp(lambda t, X: dX_dt(t, X, dynamics), [0., T_YEARS], X0, t_eval=times, method='RK45', atol=1e-9, rtol=1e-6)
    #print("Размерность решения:", sol.y.shape)
    return sol.y[:2].T  # Только XY координаты FCS

positions = simulate_motion(dynamics, T_YEARS, V_0, DT) 
plt.figure(figsize=(8, 8))
plt.plot(positions[:, 0], positions[:, 1], 'o-', label='Траектория аппарата')
plt.plot(positions[0, 0], positions[0, 1], 'o', color='red', label='Начальная точка')
plt.xlabel('X (а. е.)')
plt.ylabel('Y (а. е.)')
plt.legend()
plt.grid()
plt.title('Свободное движение аппарата в FCS за 20 лет')
plt.show()

