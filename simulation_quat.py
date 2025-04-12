from Dynamics_SGLF import Dynamics_SGLF
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
from scipy.optimize import fsolve
from scipy.spatial.transform import Rotation as R
from datetime import datetime, timedelta
import ast  # Для безопасного преобразования строки в список
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.integrate import OdeSolution, OdeSolver, RK45
from scipy.interpolate import BPoly

T_YEARS = 20.  # год
V_0 = 25.  # а.е./год, скорость аппарата в инерц системе
DT = 0.05  # Шаг для t_span
t0_data = datetime(2030, 9, 1, 0, 0)
z = 550.  # a.e. 
z0 = 10. * 63241.1  # расстояние от КА до Солнца (FCS), а.е.
t0 = 0.
time_grid = np.arange(0., T_YEARS, 0.1)  # в годах

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
    t0_data=t0_data,
    z0=z0, time_grid=time_grid)

def Dynamics(dynamics, t):
    """
    Вычисляет угл скорость, ускорение СК, ускорение Солнца.
    """
    dR_dt, d2R_dt = dynamics.r_sun_deriv(t)
    q, omega, epsilon = dynamics.get_dynamics(t)
    return q, omega, epsilon, dR_dt, d2R_dt

def dX_dt(t, X, dynamics):
    """
    Производные для интегрирования методом RK45.
    """
    q, omega, epsilon, dR_SSB, d2R_SSB = Dynamics(dynamics, t)
    q_rot = R.from_quat(q)
    q_inv = q_rot.inv()
    S, _ = dynamics.basis_FCS(t)
    dS_dt , d2S_dt = dynamics.dS_dt(t)
    # rotation_matrix = q_rot.as_matrix()
    # print("S:", S)
    # print("матр из кватерниона:", rotation_matrix)
    d2R_FCS1 = S.T @ d2R_SSB
    d2R_FCS = q_inv.apply(d2R_SSB)
    # print(d2R_FCS1)
    # print(d2R_FCS)
    # exit()
    p_FCS = X[:3]
    dpdt_FCS = X[3:]
    d2pdt2_FCS = - d2R_FCS - 2 * (np.cross(omega, dpdt_FCS)) - np.cross(epsilon, p_FCS)  - np.cross(omega, np.cross(omega, p_FCS)) 
    D1 = (d2S_dt @ p_FCS) + 2 * (dS_dt @ dpdt_FCS)
    d2pdt2_FCS1 =  - S.T @ D1 - d2R_FCS
    # print("уравнения через кват:", d2pdt2_FCS)
    # print("уравнения через матрицу:", d2pdt2_FCS1)
    # exit()
    dpdt_FCS = np.asarray(dpdt_FCS).flatten()
    d2pdt2_FCS = np.asarray(d2pdt2_FCS).flatten()
    return np.concatenate((dpdt_FCS, d2pdt2_FCS), axis=0)

def simulate_motion(dynamics, T_YEARS, V_0, DT):
    """
    Интеграция уравнений движения методом Адамса (через LSODA).
    """
    t0 = 0.
    q, omega, epsilon, dRdt_SSB_0, d2Rdt_SSB_0 = Dynamics(dynamics, t0)
    q_rot = R.from_quat(q)
    times = np.arange(0., T_YEARS, DT)
    p0_FCS = np.array([0.0, 0.0, z])  # переменная z должна быть определена заранее
    S0, _ = dynamics.basis_FCS(t0)
    dS0_dt, _ = dynamics.dS_dt(t0)
    drdt_SSB_0 = V_0 * S0[:, 2]
    dpdt_FCS_0 = S0.T @ (drdt_SSB_0 - dRdt_SSB_0 - dS0_dt @ p0_FCS)
    X0 = np.concatenate((p0_FCS, dpdt_FCS_0))

    sol = solve_ivp(
        fun=lambda t, X: dX_dt(t, X, dynamics),
        t_span=[0., T_YEARS],
        y0=X0,
        t_eval=times,
        method='LSODA',  # <--- Метод Адамса
        atol=1e-9,
        rtol=1e-6
    )

    return sol.y[:2].T  # Только XY координаты FCS


positions = 1.496 * 1e8 * simulate_motion(dynamics, T_YEARS, V_0, DT)
plt.figure(figsize=(8, 8))
plt.plot( positions[:, 0] ,  positions[:, 1] , 'o-', label='Траектория аппарата')
plt.plot(positions[0, 0],  positions[0, 1], 'o', color='green', label='Начальная точка')
plt.plot(positions[-1, 0],  positions[-1, 1], 'o', color='red', label='Конечная точка')
plt.axis('equal')
plt.xlabel('X (км)')
plt.ylabel('Y (км)')
plt.legend()
plt.grid()
plt.title('Свободное движение аппарата в FCS за 20 лет')
plt.show()
