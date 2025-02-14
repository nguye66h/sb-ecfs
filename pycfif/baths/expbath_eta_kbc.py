import numpy as np
from typing import Union

from scipy import integrate
from pycfif.eta_kbc import Eta_KBC
from pycfif.baths.expbath import ExponentialBath
from pycfif.sim_params import SimulationParamsSS, SimulationParamsKBC

def cosr(x, x0 = 1e-12):
    if np.abs(x) < x0:
        return 0.5 - (x**2 / 24.0) + (x**4 / 720)
    else:
        return (1 - np.cos(x))/(x**2)

def coshr(x, x0 = 5e-3):
    if np.abs(x) < x0:
        return - (0.5 + (x**2 / 24.0) + (x**4 / 720))
    else:
        return (1 - np.cosh(x))/(x**2)

def sinhc(x, x0 = 1e-2):
    if abs(x) < x0:
        return 1.0 + (x**2 / 6) + (x**4 / 120)
    else:
        return np.sinh(x) / x

class ExpBath_Eta_KBC(Eta_KBC):

    def __init__(self, bath: ExponentialBath, sp: Union[SimulationParamsSS, SimulationParamsKBC]):
        assert isinstance(bath, ExponentialBath)
        assert isinstance(sp, SimulationParamsKBC) or isinstance(sp, SimulationParamsSS)
        
        super().__init__(bath, sp)

    def sinsinhr(self, w, x0 = 1e-12):
        bpt = self.db + self.dt
        bmt = self.db - self.dt
        if np.abs(w) < x0:
            return (self.dt * self.db/4) * (0.25 + (bpt * bmt * w**2 / 24.0) + ((3*(bpt * bmt)**2 - 4 * self.db**2 * self.dt**2) * w**4 / 5760.0))
        else:
            return np.sin(w*self.dt/2) * np.sinh(w*self.db/2) / w**2
        
    def eta_pp_tt_kk(self, d: int) -> complex:

        x0 = 1e-12
        cut = integrate.quad(
            lambda w: 2 * self.jw(w) * cosr(w*self.dt) * np.cos(w*self.dt*d)/ np.tanh(self.b*w/2),
            0.0,x0,
            points=[0.0]
        )[0]
        
        if self.dt == 0 or self.t == 0 or d == 0:
            res_re = integrate.quad(
                lambda w: 2 * self.jw(w) * cosr(w*self.dt) / np.tanh(self.b*w/2),
                x0,np.inf
            )[0]
            return 2*(self.dt**2)*(cut+res_re + 0.0 * 1j)

        res_re = integrate.quad(
            lambda w: 2 * self.jw(w) * cosr(w*self.dt) / np.tanh(self.b*w/2),
            x0,np.inf,
            weight='cos',
            wvar=self.dt*d
        )[0]
        res_im = integrate.quad(
            lambda w: 2 * self.jw(w) * cosr(w*self.dt),
            0.00,np.inf,
            weight='sin',
            wvar=self.dt*d
        )[0]

        return (self.dt**2)*(cut+res_re + 1j * res_im)


    def eta_pm_tt_kk(self, d: int) -> complex:

        x0 = 1e-12
        cut = integrate.quad(
            lambda w: -2 * self.jw(w) * cosr(w*self.dt) * np.cos(w*self.dt*d)/ np.tanh(self.b*w/2),
            0.0,x0,
            points=[0.0]
        )[0]

        if self.dt == 0 or self.t == 0 or d == 0:
            res_re = integrate.quad(
                lambda w: -2 * self.jw(w) * cosr(w*self.dt) / np.tanh(self.b*w/2),
                x0,np.inf
            )[0]
            return 2*(self.dt**2)*(cut+res_re + 0.0 * 1j)

        res_re = integrate.quad(
            lambda w: -2 * self.jw(w) * cosr(w*self.dt) / np.tanh(self.b*w/2),
            x0,np.inf,
            weight='cos',
            wvar=self.dt*d
        )[0]
        res_im = integrate.quad(
            lambda w: 2 * self.jw(w) * cosr(w*self.dt),
            0.0,np.inf,
            weight='sin',
            wvar=self.dt*d
        )[0]
    
        return (self.dt**2)*(cut+res_re + 1j * res_im)


    def eta_pm_tt_k(self) -> complex:

        x0 = 1e-12
        cut = integrate.quad(
            lambda w: -2 * self.jw(w) * np.sinc(w*self.dt/(2*np.pi))**2 / np.tanh(self.b*w/2),
            0.0, x0,
            points=[0.0]
        )[0]
        res_re = integrate.quad(
            lambda w: -2 * self.jw(w) * np.sinc(w*self.dt/(2*np.pi))**2 / np.tanh(self.b*w/2),
            x0, np.inf
        )[0]

        return (self.dt/2)**2 * (cut+res_re + 0.0 * 1j)

    
    def eta_pp_tt_k(self) -> complex:

        x0 = 1e-12
        cut_re = integrate.quad(
            lambda w: 2 * self.jw(w) * np.sinc(w*self.dt/(2*np.pi))**2 / np.tanh(self.b*w/2),
            0.0, x0,
            points=[0.0]
        )[0]
        res_re = integrate.quad(
            lambda w: 2 * self.jw(w) * np.sinc(w*self.dt/(2*np.pi))**2 / np.tanh(self.b*w/2),
            x0, np.inf
        )[0]
        cut_im = integrate.quad(
            lambda w: self.dt * self.jw(w) * (1 - np.sinc(w*self.dt/np.pi)) / w,
            0.0, x0,
            points=[0.0]
        )[0]
        res_im = integrate.quad(
            lambda w: self.dt * self.jw(w) * (1 - np.sinc(w*self.dt/np.pi)) / w,
            x0, np.inf
        )[0]

        return (self.dt/2)**2 * (cut_re+res_re) + 1j*(cut_im+res_im)

    
    ###############

    
    def eta_pp_mix_kk(self, k: int, kp: int) -> complex:

        if self.dt == 0 or self.t == 0:
            return 0.0 + 0.0 * 1j

        x0 = 1e-12
        max_w = 75 * self.wc/(1 + self.wc * min(self.db*(self.M - (kp+1)), self.db*(self.M + kp)))
        
        if k > 1:
            res_re = integrate.quad(
                lambda w: -(4/2) * self.jw(w)  * np.sin(w*self.dt/2) * ((1 - np.exp(-w * self.db)) / (1 - np.exp(-w * self.b))) * ( np.exp(-w*self.db*(self.M - (kp+1))) - np.exp(-w*self.db*(self.M + kp)) )/(w**2),
                x0, max_w, #np.inf,
                weight='sin',
                wvar=self.dt*(k-0.5)
            )[0]
        else:
            res_re = integrate.quad(
                lambda w: -(4/2) * self.jw(w)  * (np.sin(w*self.dt/2) ** 2) * ((1 - np.exp(-w * self.db)) / (1 - np.exp(-w * self.b))) * ( np.exp(-w*self.db*(self.M - (kp+1))) - np.exp(-w*self.db*(self.M + kp)) )/(w**2),
                x0, max_w #np.inf
            )[0]
        cut_re = integrate.quad(
            lambda w: -4 * self.jw(w)  * self.sinsinhr(w) * np.sinh(w*self.db*(kp + 0.5)) * np.sin(w*self.dt*(k - 0.5))/ np.sinh(w*self.b/2),
            0.0, x0,
            points=[0.0]
        )[0]
        res_im = integrate.quad(
            lambda w: (4/2) * self.jw(w)  * np.sin(w*self.dt/2) * ((1 - np.exp(-w * self.db)) / (1 - np.exp(-w * self.b))) * ( np.exp(-w*self.db*(self.M - (kp+1))) + np.exp(-w*self.db*(self.M + kp)) )/(w**2),
            x0, np.inf,
            weight='cos',
            wvar=self.dt*(k-0.5)
        )[0]
        cut_im = integrate.quad(
            lambda w: 4 * self.jw(w)  * self.sinsinhr(w) * np.cosh(w*self.db*(kp + 0.5)) * np.cos(w*self.dt*(k - 0.5)) / np.sinh(w*self.b/2),
            0.0, x0,
            points=[0.0]
        )[0]

        return ((cut_re + res_re) - 1j*(cut_im + res_im)).conjugate()

    
    def eta_pm_mix_kk(self, k: int, kp: int) -> complex:

        if self.dt == 0 or self.t == 0:
            return 0.0 + 0.0 * 1j

        x0 = 1e-12
        max_w = 75 * self.wc/(1 + self.wc * min(self.db*(self.M - (kp+1)), self.db*(self.M + kp)))
        
        if k > 1:
            res_re = integrate.quad(
                lambda w: (4/2) * self.jw(w)  * np.sin(w*self.dt/2) * ((1 - np.exp(-w * self.db)) / (1 - np.exp(-w * self.b))) * ( np.exp(-w*self.db*(self.M - (kp+1))) - np.exp(-w*self.db*(self.M + kp)) )/(w**2),
                x0, max_w, #np.inf,
                weight='sin',
                wvar=self.dt*(k-0.5)
            )[0]
                    
        else:
            res_re = integrate.quad(
                lambda w: (4/2) * self.jw(w)  * (np.sin(w*self.dt/2) ** 2) * ((1 - np.exp(-w * self.db)) / (1 - np.exp(-w * self.b))) * ( np.exp(-w*self.db*(self.M - (kp+1))) - np.exp(-w*self.db*(self.M + kp)) )/(w**2),
                x0, max_w #np.inf
            )[0]
        cut_re = integrate.quad(
            lambda w: 4 * self.jw(w)  * self.sinsinhr(w) * np.sinh(w*self.db*(kp + 0.5)) * np.sin(w*self.dt*(k - 0.5))/ np.sinh(w*self.b/2),
            0.0, x0,
            points=[0.0]
        )[0]
        res_im = integrate.quad(
            lambda w: (4/2) * self.jw(w)  * np.sin(w*self.dt/2) * ((1 - np.exp(-w * self.db)) / (1 - np.exp(-w * self.b))) * ( np.exp(-w*self.db*(self.M - (kp+1))) + np.exp(-w*self.db*(self.M + kp)) )/(w**2),
            x0, np.inf,
            weight='cos',
            wvar=self.dt*(k-0.5)
        )[0]
        cut_im = integrate.quad(
            lambda w: 4 * self.jw(w)  * self.sinsinhr(w) * np.cosh(w*self.db*(kp + 0.5)) * np.cos(w*self.dt*(k - 0.5)) / np.sinh(w*self.b/2),
            0.0, x0,
            points=[0.0]
        )[0]
        
        return ((cut_re + res_re) - 1j*(cut_im + res_im)).conjugate()

    
    ###############


    def eta_pp_bb_kk(self, d: int) -> complex:

        x0 = 1e-3
        cut = integrate.quad(
            lambda w: 2 * self.jw(w) * (np.exp(-w*self.db*d) + (np.exp(-w*(self.b - self.db*d)) + np.exp(-w* (self.b + self.db*d)))/(1 - np.exp(-self.b*w))) * coshr(w*self.db),
            0.0, x0
        )[0] * (self.db**2)
        res_re = integrate.quad(
            lambda w: (self.jw(w) /(1 - np.exp(-self.b*w))) * (2*np.exp(-w*self.db*(2*self.M - d)) + 2*np.exp(-w*self.db*d) - np.exp(-w*self.db*(2*self.M - d - 1)) - np.exp(-w*self.db*(d - 1)) - np.exp(-w*self.db*(2*self.M - d + 1)) - np.exp(-w*self.db*(d + 1)) ) / (w**2),
            x0, np.inf
        )[0]

        return res_re + 0.0 * 1j

    
    def eta_pm_bb_kk(self, sum_kkp: int) -> complex:

        x0 = 1e-3
        
        kkp = sum_kkp
        kkp1 = sum_kkp + 1
        kkp2 = sum_kkp + 2

        cut = integrate.quad(
            lambda w: 2 * (self.jw(w) / np.sinh(w*self.b/2)) * coshr(w*self.db) * np.cosh(w * self.db * (self.M - kkp1)),
            0.0, x0,
            points=[0.0]
        )[0] * (self.db)**2
        
        res_re = integrate.quad(
            lambda w: self.jw(w) * (1/(1 - np.exp(-w*self.b))) * (2 * np.exp(-w * self.db * kkp1) + 2 * np.exp(-w*self.db*(2*self.M - kkp1)) - np.exp(-w*self.db*kkp) - np.exp(-w*self.db*(2*self.M - kkp2)) - np.exp(-w*self.db*kkp2) - np.exp(-w*self.db*(2*self.M - kkp)) ) / (w**2),
            x0, np.inf
        )[0]

        return cut + res_re + 0.0 * 1j


    def eta_pp_bb_k(self) -> complex:
    
        x0 = 1e-3
        cut = integrate.quad(
            lambda w: self.jw(w) * (self.db * coshr(w*self.db)/(np.tanh(self.b*w/2)) - (1 - sinhc(w*self.db)) / w),
            0.0, x0,
            points=[0.0]
        )[0]
        res_re = integrate.quad(
            lambda w: self.jw(w) * (-1 - (np.exp(-w*self.db) + np.exp(-w*(self.b - self.db)))/(w * self.db * (1-np.exp(-w*self.b))) + 1/(np.tanh(w*self.b/2)*w*self.db) ) / w,
            x0, np.inf
        )[0]
        
        return self.db * (cut+res_re + 0.0 * 1j)


    def eta_pm_bb_k(self, k: int) -> complex:

        return 0.5*self.eta_pm_bb_kk(2*k)


    ###################
    ###################

    def Eta_pp_tt_kk_zeroT(self, d):

        x0 = 1e-12
        cut = integrate.quad(
            lambda w: 2 * self.jw(w) * cosr(w*self.dt) * np.cos(w*self.dt*d),
            0.0,x0,
            points=[0.0]
        )[0]
        
        res_re = integrate.quad(
            lambda w: 2 * self.jw(w) * cosr(w*self.dt),
            x0,np.inf,
            weight='cos',
            wvar=self.dt*d
        )[0]
        res_im = integrate.quad(
            lambda w: 2 * self.jw(w) * cosr(w*self.dt),
            0.00,np.inf,
            weight='sin',
            wvar=self.dt*d
        )[0]

        return (self.dt**2)*(cut+res_re + 1j * res_im)
        
    def Eta_pp_tt_k_zeroT(self):

        x0 = 1e-12
        cut_re = integrate.quad(
            lambda w: 2 * self.jw(w) * np.sinc(w*self.dt/(2*np.pi))**2,
            0.0, x0,
            points=[0.0]
        )[0]
        res_re = integrate.quad(
            lambda w: 2 * self.jw(w) * np.sinc(w*self.dt/(2*np.pi))**2,
            x0, np.inf
        )[0]
        cut_im = integrate.quad(
            lambda w: self.dt * self.jw(w) * (1 - np.sinc(w*self.dt/np.pi)) / w,
            0.0, x0,
            points=[0.0]
        )[0]
        res_im = integrate.quad(
            lambda w: self.dt * self.jw(w) * (1 - np.sinc(w*self.dt/np.pi)) / w,
            x0, np.inf
        )[0]

        return (self.dt/2)**2 * (cut_re+res_re) + 1j*(cut_im+res_im)
