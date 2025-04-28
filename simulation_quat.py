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

T_YEARS = 30. / 365.25  # год
V_0 = 25.  # а.е./год, скорость аппарата в инерц системе
DT = 0.05  # Шаг для t_span 
t0_data = datetime(2030, 9, 1, 0, 0)
z = 550.  # a.e. 
z0 = 10. * 63241.1  # расстояние от КА до Солнца (FCS), а.е.
t0 = 0.
h_interp = 0.005
time_grid = np.arange(0., T_YEARS+0.01, h_interp)  # в годах

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
    z0=z0, T_YEARS=T_YEARS, h_interp=h_interp, time_grid = time_grid)

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
    q, omega_g, epsilon_g, dR_SSB, d2R_SSB = Dynamics(dynamics, t)
    q_rot = R.from_quat(q)
    q_inv = q_rot.inv()
    d2R_FCS = q_inv.apply(d2R_SSB)
    p_FCS = X[:3]
    dpdt_FCS = X[3:]
    d2pdt2_FCS = - d2R_FCS - 2 * (np.cross(omega_g, dpdt_FCS)) - np.cross(epsilon_g, p_FCS)  - np.cross(omega_g, np.cross(omega_g, p_FCS)) 
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

    p0_FCS = np.array([0.0, 0.0, z])  
    S0, _ = dynamics.basis_FCS(t0)
    dS0_dt, _ = dynamics.dS_dt(t0)
    drdt_SSB_0 = V_0 * S0[:, 2]
    dpdt_FCS_0 = S0.T @ (drdt_SSB_0 - dRdt_SSB_0 - dS0_dt @ p0_FCS)
    X0 = np.concatenate((p0_FCS, dpdt_FCS_0))
    
    sol = solve_ivp(lambda t, X: dX_dt(t, X, dynamics), [0., T_YEARS], X0, t_eval=times, method='RK45', atol=1e-9, rtol=1e-6)
    return sol.y[:2].T  # Только XY координаты FCS

positions = 1.496 * 1e8 * simulate_motion(dynamics, T_YEARS, V_0, DT)
import simulation
positions_S = simulation.positions
min_length = min(len(positions), len(positions_S))
positions_trimmed = positions[:min_length]
positions_S_trimmed = positions_S[:min_length]

diff_norm = np.linalg.norm(positions_trimmed - positions_S_trimmed, axis=1)


times = np.arange(0., T_YEARS, DT)  
times_trimmed = times[:min_length]

# #Строим график
# plt.figure(figsize=(10, 6))
# plt.plot(times_trimmed, diff_norm, label='Норма разности (по XY)', color='purple')
# plt.xlabel('Время (годы)')
# plt.ylabel('Норма разности (км)')
# plt.title('Норма разности между positions и positions_S для шага 0.01 лет и tfinal = 30 дней')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()
# exit()


# plt.figure(figsize=(8, 8))
# plt.plot( positions[:, 0] ,  positions[:, 1] , 'o-', label='Траектория аппарата')
# plt.plot(positions[0, 0],  positions[0, 1], 'o', color='green', label='Начальная точка')
# plt.plot(positions[-1, 0],  positions[-1, 1], 'o', color='red', label='Конечная точка')
# plt.axis('equal')
# plt.xlabel('X (км)')
# plt.ylabel('Y (км)')
# plt.legend()
# plt.grid()
# plt.title('Свободное движение аппарата в FCS за 20 лет c оптимизацией')
# plt.show()


# Часть 1: Подготовка
N = 10  # Количество различных начальных условий
radius_km = 100_000.0  # Радиус в км
radius_au = radius_km / 1.496e8  # Перевод в а.е.

angles = np.linspace(0, 2 * np.pi, N, endpoint=False)  # равномерные углы
positions_all = []

for angle in angles:
    # Генерация новой начальной позиции в XY (на кольце радиуса)
    x0 = radius_au * np.cos(angle)
    y0 = radius_au * np.sin(angle)
    z0_local = z  # оставляем прежний Z

    # Пересоздаем dynamics (можно и один и тот же использовать, но для ясности — копия)
    dynamics_i = Dynamics_SGLF(
        Orb_param_exo_array=Orb_param_exo_array,
        M_sun=M_sun,
        M_JSUN_array=M_JSUN_array,
        Orb_param_JSUN_array=Orb_param_JSUN_array,
        t0=t0,
        t0_data=t0_data,
        z0=z0,
        T_YEARS=T_YEARS,
        h_interp=h_interp, time_grid=time_grid)
    

    def simulate_motion_offset(x, y, dynamics):
        t0 = 0.
        q, omega, epsilon, dRdt_SSB_0, d2Rdt_SSB_0 = Dynamics(dynamics, t0)
        times = np.arange(0., T_YEARS, DT)
        S0, _ = dynamics.basis_FCS(t0)
        dS0_dt, _ = dynamics.dS_dt(t0)

        # смещённая начальная точка
        p0_FCS = np.array([x, y, z0_local])
        drdt_SSB_0 = V_0 * S0[:, 2]
        dpdt_FCS_0 = S0.T @ (drdt_SSB_0 - dRdt_SSB_0 - dS0_dt @ p0_FCS)
        X0 = np.concatenate((p0_FCS, dpdt_FCS_0))

        sol = solve_ivp(lambda t, X: dX_dt(t, X, dynamics), [0., T_YEARS], X0, t_eval=times,
                        method='RK45', atol=1e-9, rtol=1e-6)
        return sol.y[:2].T * 1.496e8  # в км

    pos_i = simulate_motion_offset(x0, y0, dynamics_i)
    positions_all.append(pos_i)

# Загрузка эталонной траектории
import simulation
positions_S = simulation.positions

# Часть 2: График норм разности
plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, N))
min_len = min([len(p) for p in positions_all] + [len(positions_S)])
times_trimmed = np.arange(0., T_YEARS, DT)[:min_len]

for i, pos in enumerate(positions_all):
    diff = np.linalg.norm(pos[:min_len] - positions_S[:min_len], axis=1)
    plt.plot(times_trimmed, diff, color=colors[i], label=f"{i}")

plt.xlabel('Время (годы)')
plt.ylabel('Норма разности (км)')
plt.title('Нормы разности траекторий для различных начальных условий (радиус 100000 км)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#_______________________________________________________________________________________________________________________________________________
# h_interp_values = [0.0001, 0.00001]  # Примеры разных шагов
# all_positions = {}


#     # Переводим угловую скорость и производную в систему g
#     # Векторная часть в кватернион -> поворот -> обратно в вектор
#     # omega_q = np.concatenate([[0], omega_i])                  # делаем из вектора кватернион [0, ωx, ωy, ωz]
#     # omega_i_rot = R.from_quat(omega_q)                        # scipy не поддерживает такие кватернионы напрямую

#     # # Применяем поворот (ручной способ)
#     # omega_g = q_inv.apply(omega_i)        # переводим omega из системы i в g
#     # epsilon_g = q_inv.apply(epsilon_i)   

#     #S, _ = dynamics.basis_FCS(t)
#     #dS_dt , d2S_dt = dynamics.dS_dt(t)
#     # rotation_matrix = q_rot.as_matrix()
#     # print("S:", S)
#     # print("матр из кватерниона:", rotation_matrix)
#     #d2R_FCS1 = S.T @ d2R_SSB

#     # sol = solve_ivp(
#     #     fun=lambda t, X: dX_dt(t, X, dynamics),
#     #     t_span=[0., T_YEARS],
#     #     y0=X0,
#     #     t_eval=times,
#     #     method='LSODA',  
#     #     atol=1e-9,
#     #     rtol=1e-6
#     # )

# for h_interp in h_interp_values:
#     dynamics = Dynamics_SGLF(
#         Orb_param_exo_array=Orb_param_exo_array,
#         M_sun=M_sun,
#         M_JSUN_array=M_JSUN_array,
#         Orb_param_JSUN_array=Orb_param_JSUN_array,
#         t0=t0,
#         t0_data=t0_data,
#         z0=z0, T_YEARS=T_YEARS, h_interp=h_interp)

#     positions = 1.496e8 * simulate_motion(dynamics, T_YEARS, V_0, DT)
#     all_positions[h_interp] = positions
# positions = 1.496 * 1e8 * simulate_motion(dynamics, T_YEARS, V_0, DT)
# import simulation
# positions_S = simulation.positions  
# plt.figure(figsize=(10, 6))

# for h_interp, positions in all_positions.items():
#     min_len = min(len(positions), len(positions_S))
#     diff = np.linalg.norm(positions[:min_len] - positions_S[:min_len], axis=1)
#     times_trimmed = np.arange(0., T_YEARS, DT)[:min_len]
#     plt.plot(times_trimmed, diff, label=f"h_interp = {h_interp}")

# plt.xlabel("Время (годы)")
# plt.ylabel("Норма разности (км)")
# plt.title("Сравнение точности при разных h_interp")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# exit()
# import simulation
# positions_S = simulation.positions
# min_length = min(len(positions), len(positions_S))
# positions_trimmed = positions[:min_length]
# positions_S_trimmed = positions_S[:min_length]

# diff_norm = np.linalg.norm(positions_trimmed - positions_S_trimmed, axis=1)


# times = np.arange(0., T_YEARS, DT)  
# times_trimmed = times[:min_length]

# # Строим график
# plt.figure(figsize=(10, 6))
# plt.plot(times_trimmed, diff_norm, label='Норма разности (по XY)', color='purple')
# plt.xlabel('Время (годы)')
# plt.ylabel('Норма разности (км)')
# plt.title('Норма разности между positions и positions_S для шага 0.01 лет')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()


# plt.figure(figsize=(6, 6))
# plt.quiver(p0[0], p0[1], v0[0], v0[1], angles='xy', scale_units='xy', scale=1, color='blue')
# plt.scatter(p0[0], p0[1], color='green', label='Начальная позиция')
# plt.xlabel('X (а.е.)')
# plt.ylabel('Y (а.е.)')
# plt.title('Фазовый вектор в начальный момент времени (FCS) c ооптимизацией')
# plt.grid()
# plt.axis('equal')
# plt.legend()
# plt.show()
