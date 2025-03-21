import numpy as np
from scipy.optimize import fsolve
from datetime import datetime

class Dynamics_SGLF():
    def __init__(self, Orb_param_exo_array, M_sun, M_JSUN_array, Orb_param_JSUN_array, t0, a_vec, z0):
        self.Orb_param_exo_array = Orb_param_exo_array # массив, содержащий орбитальные параметры экзопланеты(соотв. табл. 2)
        self.M_sun = M_sun # масса Солнца
        self.M_JSUN_array = M_JSUN_array # массив, содержащий массы планет Солнечной системы
        self.Orb_param_JSUN_array = Orb_param_JSUN_array # массив, содержащий орбитальные параметры [a, T, e, Ω, ω, i, t0] планет Солнечной системы(соотв. табл. 1)
        self.t0 = t0 # начальное время
        self.a_vec = a_vec 
        self.z0 = z0

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
       
        Returns
        -------
        r : np.ndarray
            радиус-вектор планеты
        
        '''
        a = Orb_param[0] # большая полуось
        T = Orb_param[1] # период
        e = Orb_param[2] # эксцентриситет
        Omega = Orb_param[3]
        w = Orb_param[4]
        i = Orb_param[5] # наклонение
        t0_planet = Orb_param[6]
        if isinstance(t0_planet, datetime):
            t0_planet = (t0_planet - datetime(2000, 1, 1)).total_seconds()
        if isinstance(t, datetime):
            t = (t - datetime(2000, 1, 1)).total_seconds()


        M = 2. * np.pi * (t - t0_planet) / T
        E = M + e* np.sin(M) + 0.5 * e**2 * np.sin(2*M) # e достаточно мал

        A = np.cos(Omega) * np.cos(w) - np.sin(Omega) * np.sin(w) * np.cos(i)
        B = np.sin(Omega) * np.cos(w) + np.cos(Omega) * np.sin(w) * np.cos(i)
        F = -np.cos(Omega) * np.sin(w) - np.sin(Omega) * np.cos(w) * np.cos(i)
        G = -np.sin(Omega) * np.sin(w) + np.cos(Omega) * np.cos(w) * np.cos(i)
        C = np.sin(w)*np.sin(i)
        H = np.cos(w) * np.sin(i)

        A1 = np.array([A, B, C])
        B1 = np.array([F, G, H])

        X_t = np.cos(E) - E
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
        # переход в сферич координаты
        phi = 90. + 50. * 1e-6 * (t-self.t0) # в град
        psi = 0. + 50. * 1e-6 * (t-self.t0) # в град
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
        n_bc = self.barycenter_position_SSB(t)
        r_sun = self.Sun_position_SSB(t)
        r_p_bc = self.kepler_problem(t, self.Orb_param_exo_array)
        delta = self.delta(r_p_bc, r_sun)
        n0 = self.focal_line_direction(t)

        g3 = n0 / np.linalg.norm(n0)
        g1 = (self.a_vec - np.dot(self.a_vec,g3) * g3) / np.linalg.norm(self.a_vec - np.dot(self.a_vec,g3) * g3)
        g2 = np.cross(g3, g1)
        S_matrix = np.array([g1, g2, g3])
        return S_matrix
    
    def r_derivative(self, t, Orb_param):
        '''
        Возвращает первую и вторую производные от радиус-вектора r, найденного с помощью 
        решения уравнения Кеплера
        '''
        a = Orb_param[0] # большая полуось
        T = Orb_param[1] # период
        e = Orb_param[2] # эксцентриситет
        Omega = Orb_param[3]
        w = Orb_param[4]
        i = Orb_param[5] # наклонение
        t0_planet = Orb_param[6]

        if isinstance(t0_planet, datetime):
            t0_planet = (t0_planet - datetime(2000, 1, 1)).total_seconds()
        if isinstance(t, datetime):
            t = (t - datetime(2000, 1, 1)).total_seconds()

        M = 2. * np.pi * (t - t0_planet) / T
        n = M / (t - t0_planet)
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
        dr_sun_dt = np.zeros(3)
        d2r_sun_dt = np.zeros(3)
        dR_JSUN_list_dt = []
        d2R_JSUN_list_dt = []
        for i in range(self.Orb_param_JSUN_array.shape[0]):
            dR_planet_dt, d2R_planet_dt = self.r_derivative(t, self.Orb_param_JSUN_array[i])
            dR_JSUN_list_dt.append(dR_planet_dt)
            d2R_JSUN_list_dt.append(d2R_planet_dt)
        dR_JSUN_array = np.array(dR_JSUN_list_dt)
        d2R_JSUN_array = np.array(d2R_JSUN_list_dt)
        for i in range(len(self.M_JSUN_array)):  # Перебираем массы планет
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
        r_sun = self.Sun_position_SSB(t)
        r_p_bc = self.kepler_problem(t, self.Orb_param_exo_array)
        delta = self.delta(r_p_bc, r_sun)
        d_delta_dt, d2_delta_dt = self.d_delta_dt(t)
        phi = 90. + 50. * 1e-6 * (t-self.t0) # в град
        psi = 0. + 50. * 1e-6 * (t-self.t0) # в град
        dphi_dt = 50 * 1e-6
        dpsi_dt = 50 * 1e-6
        n_bc = self.barycenter_position_SSB(t)
        dn_bc_dt = np.array([
            -np.sin(psi) * dpsi_dt * np.cos(phi) - np.sin(psi) * np.cos(phi) * dpsi_dt,
            np.cos(psi) * dpsi_dt * np.cos(phi) - np.sin(psi) * np.sin(phi) * dpsi_dt,
            np.cos(phi) * dphi_dt])

        d2n_bc_dt = np.array([
            -2 * np.cos(psi) * np.cos(phi) * dpsi_dt**2 + 2 * np.sin(psi) * np.sin(phi) * dphi_dt * dpsi_dt,
            np.cos(psi) * np.cos(phi) - (np.sin(psi) * np.cos(phi) + np.cos(psi) * np.sin(phi)) * dpsi_dt**2 - np.cos(psi) * np.sin(phi) * dphi_dt * dpsi_dt,
            - np.sin(phi) * dphi_dt**2])


        #delta_normal = delta - np.dot(n_bc, delta) * n_bc
        d_delta_normal_dt = d_delta_dt - np.dot(dn_bc_dt, delta) * n_bc - np.dot(n_bc, d_delta_dt) * n_bc - np.dot(n_bc, delta) * dn_bc_dt
        d2_delta_normal_dt = (
                    d2_delta_dt 
                    - np.dot(d2n_bc_dt, delta) * n_bc
                    - np.dot(n_bc, d2_delta_dt) * n_bc
                    - np.dot(n_bc, delta) * d2n_bc_dt
                    - 2 * (
                        np.dot(dn_bc_dt, d_delta_dt) * n_bc 
                        + np.dot(dn_bc_dt, delta) * dn_bc_dt 
                        + np.dot(n_bc, d_delta_dt) * dn_bc_dt
                    )
                )
        
        dn0_dt = - dn_bc_dt - d_delta_normal_dt / self.z0
        d2n0_dt = - d2n_bc_dt - d2_delta_normal_dt / self.z0
        return dn0_dt, d2n0_dt
    
    def dS_dt(self, t):
        '''
        Функция возвращает первую и вторую производные матрицы S , составленной из базисных векторов g1, g2, g3 системы FCS
        
        '''
        n0 = self.focal_line_direction(t)
        d_n0_dt, d2_n0_dt = self.d_n0_dt(t)
        r_sun = self.Sun_position_SSB(t)
        r_p_bc = self.kepler_problem(t, self.Orb_param_exo_array)
        delta = self.delta(r_p_bc, r_sun)
        n_bc = self.barycenter_position_SSB(t)

        S_matrix = self.basis_FCS(t)
        g1 = S_matrix[0]
        g2 = S_matrix[1]
        g3 = S_matrix[2]

        d_g3_dt = (d_n0_dt - np.dot(g3, d_n0_dt)) / np.linalg.norm(n0)
        d2_g3_dt = (d2_n0_dt - np.dot(g3, d2_n0_dt) * g3 - 2 * np.dot(g3, d_n0_dt) * d_g3_dt 
                    - np.dot(d_g3_dt, d_n0_dt) * g3 ) / np.linalg.norm(n0)

        a_normal = - np.dot(self.a_vec,g3) * g3 + self.a_vec
        d_a_normal_dt =  - np.dot(self.a_vec,d_g3_dt) * g3 - np.dot(self.a_vec,g3) * d_g3_dt
        d2_a_normal_dt = - np.dot(self.a_vec,d2_g3_dt) * g3 - 2 * np.dot(self.a_vec,d_g3_dt) * d_g3_dt - np.dot(self.a_vec,g3) * d2_g3_dt
        
        d_g1_dt = (a_normal - np.dot(g1, d_a_normal_dt) * g1) / np.linalg.norm(a_normal)
        d2_g1_dt = (d2_a_normal_dt - np.dot(g1, d2_a_normal_dt) * g1 - 2 * np.dot(g1, d_a_normal_dt) * d_g1_dt 
                    - np.dot(d_g1_dt, d_a_normal_dt) * g1 ) / np.linalg.norm(a_normal)
        
        d_g2_dt = np.cross(d_g3_dt, g1) + np.cross(g3, d_g1_dt)
        d2_g2_dt = np.cross(d2_g3_dt, g1) + 2 * np.cross(d_g3_dt, d_g1_dt) + np.cross(g3, d2_g1_dt) 

        dS_dt = np.array([d_g1_dt, d_g2_dt, d_g3_dt])
        d2S_dt = np.array([d2_g1_dt, d2_g2_dt, d2_g3_dt])
        return dS_dt, d2S_dt

    
    
   
    