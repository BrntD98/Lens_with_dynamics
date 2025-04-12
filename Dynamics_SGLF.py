import numpy as np
from scipy.optimize import fsolve
from scipy.spatial.transform import Rotation as R
from datetime import datetime, timedelta
from scipy.spatial.transform import Rotation as R, Slerp

class Dynamics_SGLF():
    def __init__(self, Orb_param_exo_array, M_sun, M_JSUN_array, Orb_param_JSUN_array, t0, t0_data, z0, time_grid):
        self.Orb_param_exo_array = Orb_param_exo_array # массив, содержащий орбитальные параметры экзопланеты(соотв. табл. 2) [а.е., год, -, град, град, град, дата]
        self.M_sun = M_sun # масса Солнца в кг
        self.M_JSUN_array = M_JSUN_array # массив, содержащий массы планет Солнечной системы в кг
        self.Orb_param_JSUN_array = Orb_param_JSUN_array # массив, содержащий орбитальные параметры [a, T, e, Ω, ω, i, t0] планет Солнечной системы([а.е., год, -, град, град, град, дата])
        self.t0 = t0 # начальное время (дата) 
        self.z0 = z0 # а. е.
        self.t0_data = t0_data # дата
        self.time_grid = time_grid
        self.precomput = self.precompute()

    def kepler_problem(self, t, Orb_param):
        '''
        Функция для рассчета радиус-вектора планеты в некоторый момент времени t 
        Уравнение Кеплера:
        n * (t - t0) = E - e* sinE -> находим E -> подставляет в (А27)
        n = 2*pi / T

        Parameters
        ----------
        t : float
            Текущий момент времени 
        Orb_param : np.ndarray
            Параметры орбиты
        углы подаются в градусах
        Returns
        -------
        r : np.ndarray  в единицах  a
            радиус-вектор планеты
        
        '''
        a = Orb_param[0] # большая полуось
        T = Orb_param[1] # период
        e = Orb_param[2] # эксцентриситет
        Omega = Orb_param[3] * np.pi / 180
        w = Orb_param[4] * np.pi / 180
        i = Orb_param[5] * np.pi / 180 # наклонение
        t0_planet = Orb_param[6] # в дата 
        if isinstance(t0_planet, datetime):
            t0_planet = (t0_planet - self.t0_data).total_seconds()
            t0_planet /= 3600. * 24. * 365.25
        if isinstance(t, datetime):
            t = (t - self.t0_data).total_seconds()
            t /= 3600. * 24. * 365.25

        # M = 2. * np.pi * ((t - t0_planet) / T - np.floor((t - t0_planet) / T))
        M = 2. * np.pi * ((t - t0_planet) % T) / T
        E = M + e* np.sin(M) + 0.5 * e**2 * np.sin(2*M) # e достаточно мал

        A = np.cos(Omega) * np.cos(w) - np.sin(Omega) * np.sin(w) * np.cos(i)
        B = np.sin(Omega) * np.cos(w) + np.cos(Omega) * np.sin(w) * np.cos(i)
        F = -np.cos(Omega) * np.sin(w) - np.sin(Omega) * np.cos(w) * np.cos(i)
        G = -np.sin(Omega) * np.sin(w) + np.cos(Omega) * np.cos(w) * np.cos(i)
        C = np.sin(w) * np.sin(i)
        H = np.cos(w) * np.sin(i)

        A1 = np.array([A, B, C])
        B1 = np.array([F, G, H])

        X_t = np.cos(E) - e
        Y_t = np.sqrt(1 - e**2) * np.sin(E)

        r = a * (A1 * X_t + B1 * Y_t)
        return r

    def Sun_position_SSB(self, t):
        '''
        Функция для расчета положения Солнца по формуле: 
        r_sun(t) = - sum (mi/ mi + M_sun)*ri
        mi - i элемент M_JSUN_array (масса одной из планет СС)
        ri - i элемент Orb_param_JSUN_array (радиус вектор одной из СС)
        
        '''
        r_sun = np.zeros(3)
        R_JSUN_list = []
        for i in range(self.Orb_param_JSUN_array.shape[0]):
            R_planet = self.kepler_problem(t, self.Orb_param_JSUN_array[i])
            R_JSUN_list.append(R_planet)
        R_JSUN_array = np.array(R_JSUN_list)

         # Рассчитываем положение Солнца
        for i in range(len(self.M_JSUN_array)):  # Перебираем массы планет
            mi = self.M_JSUN_array[i]
            r_sun -= (mi / (mi + self.M_sun)) * R_JSUN_array[i]

        return r_sun

    def barycenter_position_SSB(self, t):
        '''
        Вычисляет направление на положение барицентра экзосистемы в SSB
        '''
        if isinstance(t, datetime):
            t = (t - self.t0_data).total_seconds()
            t /= 3600. * 24. * 365.25

        phi = (90. + 1.39 * 1e-8 * t ) * np.pi / 180. # в радианах
        psi = (0. + 1.39 * 1e-8 *  t) * np.pi / 180. # в радианах

        n_bc = np.array([np.cos(psi)*np.cos(phi),
                         np.sin(psi)*np.cos(phi),
                         np.sin(phi)])
        return n_bc
    
    @staticmethod
    def delta(r_p_bc, r_sun):
        return r_p_bc - r_sun
    
    def focal_line_direction(self, t):
        n_bc = self.barycenter_position_SSB(t)
        r_p_bc = self.kepler_problem(t, self.Orb_param_exo_array)
        r_sun = self.Sun_position_SSB(t)
        delta = self.delta(r_p_bc, r_sun)
        n0 = - n_bc - (delta - np.dot(delta, n_bc) * n_bc)/self.z0
        return n0

    def basis_FCS(self, t):
        '''
        Вычисляет базис FCS , те находит вектора g1, g2, g3

        '''
        #n_bc = self.barycenter_position_SSB(t)
        #r_sun = self.Sun_position_SSB(t)
        #r_p_bc = self.kepler_problem(t, self.Orb_param_exo_array)
        #delta = self.delta(r_p_bc, r_sun)
        n0 = self.focal_line_direction(t)
        g3 = n0 / np.linalg.norm(n0)
        # if g3[0] != 1:
        #     a_vec =  np.array([1, 0, 0])
        # elif g3[1] != 1:
        #     a_vec =  np.array([0, 1, 0])
        # else:
        #     a_vec =  np.array([0, 0, 1])
        a_vec = np.array([1., 0., 0.])
        buf = a_vec - np.dot(a_vec,g3) * g3
        g1 = buf / np.linalg.norm(buf)
        g2 = np.cross(g3, g1)
        S_matrix = np.array([g1, g2, g3]).T
        return S_matrix, a_vec
    
    def r_derivative(self, t, Orb_param):
        '''
        Возвращает первую и вторую производные от радиус-вектора r, найденного с помощью 
        решения уравнения Кеплера 
        '''
        a = Orb_param[0] # большая полуось, ае
        T = Orb_param[1] # период, год
        e = Orb_param[2] # эксцентриситет
        Omega = Orb_param[3] * np.pi / 180
        w = Orb_param[4] * np.pi / 180
        i = Orb_param[5] * np.pi / 180 # наклонение
        t0_planet = Orb_param[6] # подается дата
        if isinstance(t0_planet, datetime):
            t0_planet = (t0_planet - self.t0_data).total_seconds()
            t0_planet /= 3600. * 24. * 365.25
        if isinstance(t, datetime):
            t = (t - self.t0_data).total_seconds()
            t /= 3600. * 24. * 365.25

        # M = 2. * np.pi * ((t - t0_planet) / T - np.floor((t - t0_planet) / T))
        delta_t = (t - t0_planet) % T
        M = 2. * np.pi * delta_t / T
        n = 2 * np.pi / T
        E = M + e* np.sin(M) + 0.5 * e**2 * np.sin(2*M) # e достаточно мал

        A = np.cos(Omega) * np.cos(w) - np.sin(Omega) * np.sin(w) * np.cos(i)
        B = np.sin(Omega) * np.cos(w) + np.cos(Omega) * np.sin(w) * np.cos(i)
        F = -np.cos(Omega) * np.sin(w) - np.sin(Omega) * np.cos(w) * np.cos(i)
        G = -np.sin(Omega) * np.sin(w) + np.cos(Omega) * np.cos(w) * np.cos(i)
        C = np.sin(w)*np.sin(i)
        H = np.cos(w) * np.sin(i)

        A1 = np.array([A, B, C])
        B1 = np.array([F, G, H])

        dr_dt = n * a /(1 - e * np.cos(E)) * (-np.sin(E) * A1 + np.sqrt(1 - e**2) * np.cos(E) * B1)
        d2r_dt2 = -n**2 * a /(1 - e * np.cos(E))**3 * ((np.cos(E) - e) * A1 + np.sqrt(1 - e**2) * np.sin(E) * B1)

        return dr_dt, d2r_dt2

    def r_sun_deriv(self, t):
        """
        Вычисляет первую и вторую производные положения Солнца в инерц системе с учетом влияния планет-гигантов.
        """
        dr_sun_dt = np.zeros(3)
        d2r_sun_dt = np.zeros(3)
        dR_JSUN_list_dt = []
        d2R_JSUN_list_dt = []
        for i in range(len(self.Orb_param_JSUN_array)):
            dR_planet_dt, d2R_planet_dt = self.r_derivative(t, self.Orb_param_JSUN_array[i])
            dR_JSUN_list_dt.append(dR_planet_dt)
            d2R_JSUN_list_dt.append(d2R_planet_dt)

        dR_JSUN_array = np.array(dR_JSUN_list_dt)
        d2R_JSUN_array = np.array(d2R_JSUN_list_dt)

        for i in range(len(self.M_JSUN_array)):  
            mi = self.M_JSUN_array[i]
            dr_sun_dt -= (mi / (mi + self.M_sun)) * dR_JSUN_array[i]
            d2r_sun_dt -= (mi / (mi + self.M_sun)) * d2R_JSUN_array[i]

        return dr_sun_dt, d2r_sun_dt

    def d_delta_dt(self,t):
        '''
        Вычисляет производную d_delta(t) / dt = d(r_p_bc - r_sun)/dt
        '''
        dr_p_bc, d2r_p_bc = self.r_derivative(t, self.Orb_param_exo_array)
        dr_sun_dt, d2r_sun_dt = self.r_sun_deriv(t)

        d_delta_dt = dr_p_bc - dr_sun_dt
        d2_delta_dt = d2r_p_bc - d2r_sun_dt
        return d_delta_dt, d2_delta_dt
    
    def d_n0_dt(self, t):
        '''
        Функция для вычисления производной n0 (вектора направления фокальной линии)
        '''
        if isinstance(t, datetime):
            t = (t - self.t0_data).total_seconds()
            t /= 3600. * 24. * 365.25
        r_sun = self.Sun_position_SSB(t)
        r_p_bc = self.kepler_problem(t, self.Orb_param_exo_array)
        delta = self.delta(r_p_bc, r_sun)
        d_delta_dt, d2_delta_dt = self.d_delta_dt(t)
        phi = (90. + 1.39 * 1e-8 * t ) * np.pi / 180. # в радианах
        psi = (0. +1.39 * 1e-8 *  t ) * np.pi / 180. # в радианах
        dphi_dt = 1.39* 1e-8 * np.pi / 180. # в радианах
        dpsi_dt = 1.39* 1e-8 * np.pi / 180. # в радианах
        n_bc = self.barycenter_position_SSB(t)
        dn_bc_dt = np.array([-np.sin(psi) * dpsi_dt * np.cos(phi) - np.cos(psi) * np.sin(phi) * dphi_dt,
            np.cos(psi) * dpsi_dt * np.cos(phi) - np.sin(psi) * np.sin(phi) * dphi_dt,
            np.cos(phi) * dphi_dt])

        d2n_bc_dt = np.array([
            - np.cos(psi) * np.cos(phi) * dpsi_dt**2 
                + np.sin(psi) * np.sin(phi) * dpsi_dt * dphi_dt
                + np.sin(psi) * np.sin(phi) * dphi_dt * dpsi_dt 
                - np.cos(psi) * np.cos(phi) * dphi_dt * dphi_dt,

            -np.sin(psi) * np.cos(phi) * dpsi_dt * dpsi_dt
                - np.cos(psi) * np.sin(phi) * dpsi_dt * dphi_dt
                - np.cos(psi) * np.sin(phi) * dphi_dt * dpsi_dt 
                - np.sin(psi) * np.cos(phi) * dphi_dt * dphi_dt,

            np.sin(phi) * dphi_dt * dphi_dt])

        #delta_normal = delta - np.dot(n_bc, delta) * n_bc
        d_delta_normal_dt = d_delta_dt - np.dot(dn_bc_dt, delta) * n_bc - np.dot(n_bc, d_delta_dt) * n_bc - np.dot(n_bc, delta) * dn_bc_dt
        d2_delta_normal_dt = (d2_delta_dt 
                - np.dot(n_bc, d2_delta_dt) * n_bc 
                - np.dot(n_bc, delta) * d2n_bc_dt 
                - np.dot(d2n_bc_dt, delta) * n_bc 
                - 2 * np.dot(dn_bc_dt, d_delta_dt) * n_bc 
                - 2 * np.dot(n_bc, d_delta_dt) * dn_bc_dt 
                - 2 * np.dot(d2n_bc_dt, delta) * dn_bc_dt )
        
        dn0_dt = - dn_bc_dt - d_delta_normal_dt / self.z0
        d2n0_dt = - d2n_bc_dt - d2_delta_normal_dt / self.z0
        return dn0_dt, d2n0_dt
    
    def dS_dt(self, t):
        '''
        Функция возвращает первую и вторую производные матрицы S , составленной из базисных векторов g1, g2, g3 системы FCS
        
        '''
        n0 = self.focal_line_direction(t)
        d_n0_dt, d2_n0_dt = self.d_n0_dt(t)
        #r_sun = self.Sun_position_SSB(t)
        #r_p_bc = self.kepler_problem(t, self.Orb_param_exo_array)
        # delta = self.delta(r_p_bc, r_sun)
        # n_bc = self.barycenter_position_SSB(t)

        S_matrix = (self.basis_FCS(t)[0])
        g1 = S_matrix[:, 0]
        g2 = S_matrix[:, 1]
        g3 = S_matrix[:, 2]

        d_g3_dt = (d_n0_dt - np.dot(g3, d_n0_dt)* g3) / np.linalg.norm(n0)
        d2_g3_dt = (d2_n0_dt - np.dot(g3, d2_n0_dt) * g3 - 2 * np.dot(g3, d_n0_dt) * d_g3_dt 
            - np.dot(g3, d_n0_dt) * d_g3_dt) / np.linalg.norm(n0)

        a_vec = self.basis_FCS(t)[1]
        a_normal = - np.dot(a_vec,g3) * g3 + a_vec
        d_a_normal_dt =  - np.dot(a_vec,d_g3_dt) * g3 - np.dot(a_vec,g3) * d_g3_dt
        d2_a_normal_dt = - np.dot(a_vec,d2_g3_dt) * g3 - 2 * np.dot(a_vec,d_g3_dt) * d_g3_dt - np.dot(a_vec,g3) * d2_g3_dt
        
        d_g1_dt = (d_a_normal_dt - np.dot(g1, d_a_normal_dt) * g1) / np.linalg.norm(a_normal)
        d2_g1_dt = (d2_a_normal_dt - np.dot(g1, d2_a_normal_dt) * g1 - 2 * np.dot(g1, d_a_normal_dt) * d_g1_dt 
            - np.dot(g1, d2_a_normal_dt) * g1 ) / np.linalg.norm(a_normal)

        
        d_g2_dt = np.cross(d_g3_dt, g1) + np.cross(g3, d_g1_dt)
        d2_g2_dt = np.cross(d2_g3_dt, g1) + 2 * np.cross(d_g3_dt, d_g1_dt) + np.cross(g3, d2_g1_dt) 

        dS_dt = np.array([d_g1_dt, d_g2_dt, d_g3_dt]).T
        d2S_dt = np.array([d2_g1_dt, d2_g2_dt, d2_g3_dt]).T
        return dS_dt, d2S_dt
    
    def angular_velocity(self, t):
        '''
        Вычисление угловой скорости вращения неинерц СК FCS относительно инерциальной
        '''
        S, _ = self.basis_FCS(t)
        dS, _ = self.dS_dt(t)
        omega= 0.5 * (np.cross(S[:,0],dS[:,0]) + np.cross(S[:,1],dS[:,1]) + np.cross(S[:,2],dS[:,2]))
        return omega
    
    def angular_acceleration(self, t):
        '''
        Вычисление углового ускорения вращения неинерц СК FCS относительно инерциальной
        '''
        S, _ = self.basis_FCS(t)
        _, d2S = self.dS_dt(t)
        epsilon =  0.5 * (np.cross(S[:,0],d2S[:,0]) + np.cross(S[:,1],d2S[:,1]) + np.cross(S[:,2],d2S[:,2]))
        return epsilon  
    
    def precompute(self):
        '''
        Метод для подготовки precomput
        '''
        precomput = {}
        for t in self.time_grid:
            S, _ = self.basis_FCS(t)
            quat = R.from_matrix(S).as_quat()
            omega = self.angular_velocity(t)
            epsilon = self.angular_acceleration(t)
            precomput[t] = {"quat": quat, "omega": omega, "epsilon": epsilon}
        return precomput
    
    def get_dynamics(self, t):
        if isinstance(t, datetime):
            t = (t - self.t0_data).total_seconds() / (3600. * 24. * 365.25)

        # # Защита от выхода за границы
        # if t < self.time_grid[0] or t > self.time_grid[-1]:
        #     raise ValueError(f"Время t={t} выходит за допустимые пределы [{self.time_grid[0]}, {self.time_grid[-1]}]")

        # Проверка находится ли t точно на узле сетки
        if np.isclose(t, self.time_grid).any():
            t_exact = self.time_grid[np.isclose(t, self.time_grid)][0]
            q = self.precomput[t_exact]["quat"]
            omega = self.precomput[t_exact]["omega"]
            epsilon = self.precomput[t_exact]["epsilon"]
            return q, omega, epsilon

        # Интерполяция
        idx = np.searchsorted(self.time_grid, t) - 1
        idx = max(0, min(idx, len(self.time_grid) - 2))
        t_k, t_k1 = self.time_grid[idx], self.time_grid[idx + 1]
        tau = (t - t_k) / (t_k1 - t_k)

        q_k = R.from_quat([self.precomput[t_k]["quat"]])
        q_k1 = R.from_quat([self.precomput[t_k1]["quat"]])

        q_k_conj = q_k.inv()  # Находим сопряженные кватернионы
        q_diff = q_k_conj * q_k1  # Вычисляем разницу
        q_diff_tau = q_diff ** tau  # Возводим разницу в степень tau
        q_interp = q_k * q_diff_tau  # Перемножаем исходный кватернион с результатом

        # Интерполяция угловых скоростей omega и других параметров
        omega_k = self.precomput[t_k]["omega"]
        omega_k1 = self.precomput[t_k1]["omega"]
        omega_interp = omega_k + tau * (omega_k1 - omega_k)

        epsilon_k = self.precomput[t_k]["epsilon"]
        epsilon_k1 = self.precomput[t_k1]["epsilon"]
        epsilon_interp = epsilon_k + tau * (epsilon_k1 - epsilon_k)

        return q_interp.as_quat(), omega_interp, epsilon_interp


if __name__ == "__main__":
    M_sun = 1.989 * 1e30  # масса Солнца (кг)
    M_JSUN_array = np.array([
        1.8982e27,  # Юпитер
        5.6834e26,  # Сатурн
        8.6810e25,  # Уран
        1.02413e26  # Нептун
    ])
    import matplotlib.pyplot as plt
    t0 = datetime(2030, 9 , 1)
    z0 = 550.
    T_years = 100.
    DT = 0.5
    t0_data = datetime(2030, 9, 1, 0, 0)
    time_grid = np.arange(0., 100., 1.)  # в годах
    Orb_param_exo_array = np.array([
        1. ,  # большая полуось ()
        1. ,  # период обращения (годы)
        0.0167086,  # эксцентриситет
        -240., 352.5, 37.5,  # аргументы орбиты
        datetime(2021, 1, 2)  # начальное время
    ])
    # Орбитальные параметры планет-гигантов
    Orb_param_JSUN_array = np.array([
        [5.2044 , 11.862 , 0.0489, 100.464, 273.867, 1.303, datetime(2023, 1, 21)],
        [9.5826 , 29.4571 , 0.0565, 113.665, 339.392, 2.485, datetime(2032, 11, 29)],
        [19.2184 , 84.0205 , 0.046381, 74.006, 96.998857, 0.773, datetime(2050, 8, 19)],
        [30.07, 164.8 , 0.008678, 131.784, 276.336, 1.767957, datetime(2042, 9, 4)]
    ])

    times = np.arange(0., T_years, DT)
    Dynamics = Dynamics_SGLF(Orb_param_exo_array, M_sun, M_JSUN_array, Orb_param_JSUN_array, t0, t0_data, z0, time_grid)
    
    r_sun_array = []
    r_sun_deriv = []
    d2_rsun_array = []

    for t in times:
        r_sun = Dynamics.Sun_position_SSB(t)
        r_sun_array.append(r_sun)
        r_deriv = Dynamics.r_sun_deriv(t)[0]
        r_sun_deriv.append(r_deriv)
        d2_r_sun = Dynamics.r_sun_deriv(t)[1]
        d2_rsun_array.append(d2_r_sun)
        r_p_bc = Dynamics.kepler_problem(t, Orb_param_exo_array)
        print(np.linalg.norm(r_p_bc))
        

    # # a T e W w i t0
    # buf = Dynamics.kepler_problem(0, np.array([1, 1, 0., 0., 0., 0., datetime(2042, 9, 4)]))
    # print(buf)

    r_sun_array = np.array(r_sun_array)
    r_sun_deriv = np.array(r_sun_deriv)
    d2_rsun_array = np.array(d2_rsun_array)
    #print(np.linalg.norm(d2_rsun_array))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Траектория
    ax1.plot(r_sun_array[:, 0], r_sun_array[:, 1], 'b-')
    ax1.plot(r_sun_array[0, 0], r_sun_array[0, 1], 'go')
    ax1.set_xlabel('X, а.е.')
    ax1.set_ylabel('Y, а.е.')
    ax1.set_title('Траектория Солнца относительно барицентра Солнечной системы')
    ax1.grid(True)
    ax1.axis('equal')

    ax2.plot(r_sun_deriv[:, 0] * 4740.47 , r_sun_deriv[:, 1] * 4740.47 , 'b-')
    ax2.plot(r_sun_deriv[0, 0] * 4740.47 , r_sun_deriv[0, 1] * 4740.47 , 'go')
    ax2.set_xlabel('X, м/с')
    ax2.set_ylabel('Y, м/с')
    ax2.set_title('Скорость Солнца')
    ax2.grid(True)
    ax2.axis('equal')

    # ax3.plot(d2_rsun_array[:, 0], d2_rsun_array[:, 1], 'b-')
    # ax3.plot(d2_rsun_array[0, 0], d2_rsun_array[0, 1], 'go')
    # ax3.set_xlabel('X, а.е. / год*год')
    # ax3.set_ylabel('Y, а.е. / год*год')
    # ax3.set_title('Ускорение Солнца')
    # ax3.grid(True)
    # ax3.axis('equal')

    plt.tight_layout()
    plt.show()
