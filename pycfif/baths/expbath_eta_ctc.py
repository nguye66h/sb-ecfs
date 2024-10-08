# TODO: REWRITE USING GAUSS-LAGUERRE QUADRATURE

import numpy as np
from scipy import integrate
from scipy import special
from pycfif.eta_ctc import Eta_CTC
from pycfif.baths.expbath import ExponentialBath
from pycfif.sim_params import SimulationParamsCTC

class ExpBath_Eta_CTC(Eta_CTC):

    def __init__(self, bath: ExponentialBath, sp: SimulationParamsCTC):
        assert isinstance(bath, ExponentialBath)
        assert isinstance(sp, SimulationParamsCTC)

        super().__init__(bath, sp)

    def eta_k(self, k: int) -> complex:

        dt = self.dt
        db = self.db

        if k <= self.N:
            diff = -self.dt - 1j* self.db
        else:
            diff =  self.dt - 1j* self.db
            dt *= -1

        # want x0 * abs(diff) << 1
        x0 = 1e-3 / np.abs(diff)

        
        cut_re = integrate.quad(
            lambda w: self.jw(w) * (1 + np.exp(-self.b * w)) * (1 - (w*diff)**2 / 12) / (1 - np.exp(-self.b * w)),
            0.0,x0,
            points=[0.0]
        )[0] * (1/2) * (diff**2)

        cut_im = integrate.quad(
            lambda w: (-w * self.jw(w)) * ( 1 - (w*diff)**2 / 20 ),
            0.0,x0,
            points=[0.0]
        )[0] * (1/6) * (diff**3)

        mid_re = integrate.quad(
            lambda w: (self.jw(w) / w**2) * (1 + np.exp(-self.b * w)) * (1 - np.cos(w*dt) * np.cosh(w*db)) / (1 - np.exp(-self.b * w)),
            x0, 1.0
        )[0]
        mid_re += integrate.quad(
            lambda w: (self.jw(w) / w**2) * (np.cos(w*dt) * np.sinh(w*db)),
            x0, 1.0
        )[0]
        
        reorg = integrate.quad(
            lambda w: self.jw(w) / w,
            x0, np.inf
        )[0]

        if dt == 0.0:
            res_re = integrate.quad(
                lambda w: -(self.jw(w) / w**2) * (np.exp(-w*db) + np.exp(-w*(self.b - db))) / (1 - np.exp(-self.b * w)),
                1.0, np.inf
            )[0]
        else:
            res_re = integrate.quad(
                lambda w: -(self.jw(w) / w**2) * (np.exp(-w*db) + np.exp(-w*(self.b - db))) / (1 - np.exp(-self.b * w)),
                1.0, np.inf,
                weight='cos',
                wvar=dt
            )[0]
        res_re += integrate.quad(
            lambda w: (self.jw(w) / w**2) * (1 + np.exp(-self.b * w)) / (1 - np.exp(-self.b * w)),
            1.0, np.inf
        )[0]
        res_re -= (db * reorg)

        res_im = integrate.quad(
            lambda w: -(self.jw(w) / w**2) * (np.exp(-w*db) - np.exp(-w*(self.b - db))) / (1 - np.exp(-self.b * w)),
            x0,np.inf,
            weight='sin',
            wvar=dt
        )[0]
        res_im += (dt * reorg)

        return cut_re + mid_re + res_re + 1j*(cut_im + res_im)


    def eta_kk(self, k: int, kp:int):

        dt = self.dt
        db = self.db
        x0 = 1e-5
        
        ## if k < kp, swap them
        if k < kp:
            kdum = k
            k = kp
            kp = kdum

        tk1 = self.tau_converter(k)
        tk = self.tau_converter(k-1)

        tkp1 = self.tau_converter(kp)
        tkp = self.tau_converter(kp-1)

        dtkp = np.real(tkp1-tkp)
        dtk = np.real(tk1-tk)

        dkkp = np.abs(kp-self.N) - np.abs(k-self.N) + np.abs(kp-1-self.N) - np.abs(k-1-self.N)

        res = 0.0 + 0.0*1j
        cut = 0.0 + 0.0*1j
        
        exponent = np.minimum(db*(k-kp-1), self.b-db*(k-kp+1)) + (1/self.wc)
        ulim = 50*(1/exponent)

        npts = 75
        
        rts, wts = special.roots_legendre(npts)
        omega_grid = ((rts + 1) * (ulim - x0) / 2) + x0
        scale_factor = (ulim - x0) / 2

        jvec = np.array([scale_factor * wts[i] * self.jw(omega_grid[i])/omega_grid[i] for i in range(0, len(rts))])
        
        if kp + k == 2*self.N+1:
            temp = integrate.quad(
                lambda w: self.jw(w) * (1 - 0.5*db*w + (4*db**2 - dtk**2 - 1j*2*db*dtk) * w**2 / 24) * (1 - 0.5*db*w + (4*db**2 - dtkp**2 - 1j*2*db*dtkp) * w**2 / 24) * ( np.exp(-w*db*(k-kp-1)) + np.exp(-w*(self.b-db*(k-kp+1))) ) / (1-np.exp(-w*self.b)),
                0.0,x0,
                points=[0.0]
            )[0]
            cut = temp * (dtk + 1j*db) * (dtkp + 1j*db)

            kernel = [((2*np.exp(-w*db)*np.cos(w*self.dt) - (1 + np.exp(-2*w*db))) / w) * ( np.exp(-w*db*(k-kp-1)) + np.exp(-w*(self.b-db*(k-kp+1))) ) / (1-np.exp(-w*self.b)) for w in omega_grid]
            res_re = np.dot(jvec, kernel)

            res = (res_re + 0.0*1j) + cut
        else:
            temp1 = integrate.quad(
                lambda w: self.jw(w) * (1 - 0.5*db*w + (4*db**2 - dtk**2 - 1j*2*db*dtk) * w**2 / 24) * (1 - 0.5*db*w + (4*db**2 - dtkp**2 - 1j*2*db*dtkp) * w**2 / 24) * np.cos(w*dt*dkkp/2) * ( np.exp(-w*db*(k-kp-1)) + np.exp(-w*(self.b-db*(k-kp+1))) ) / (1-np.exp(-w*self.b)),
                0.0,x0,
                points=[0.0]
            )[0]
            temp2 = integrate.quad(
                lambda w: self.jw(w) * (1 - 0.5*db*w + (4*db**2 - dtk**2 - 1j*2*db*dtk) * w**2 / 24) * (1 - 0.5*db*w + (4*db**2 - dtkp**2 - 1j*2*db*dtkp) * w**2 / 24) * np.sin(w*dt*dkkp/2) * ( np.exp(-w*db*(k-kp-1)) - np.exp(-w*(self.b-db*(k-kp+1))) ) / (1-np.exp(-w*self.b)),
                0.0,x0,
                points=[0.0]
            )[0]
            cut = (temp1 + 1j*temp2) * (dtk + 1j*db) * (dtkp + 1j*db)
            
            if dtk*dtkp > 0.0:
                kern_cos = np.cos(omega_grid * self.dt*dkkp/2)
                kern_sin = np.sin(omega_grid * self.dt*dkkp/2)
                kern_part1 = [((2*np.exp(-w*db) - (1 + np.exp(-2*w*db)) * np.cos(w*dt))/w) for w in omega_grid]
                kern_part2 = [((1 - np.exp(-2*w*db))/w) * np.sin(w*(dtk+dtkp)/2) for w in omega_grid]
                
                kernel_re = [( np.exp(-w*db*(k-kp-1)) + np.exp(-w*(self.b-db*(k-kp+1))) ) / (1-np.exp(-w*self.b)) for w in omega_grid]
                res_re1 = np.dot(jvec, kern_cos * kern_part1 * kernel_re)

                kernel_im = [( np.exp(-w*db*(k-kp-1)) - np.exp(-w*(self.b-db*(k-kp+1))) ) / (1-np.exp(-w*self.b)) for w in omega_grid]
                res_im2 = np.dot(jvec, kern_sin * kern_part1 * kernel_im)
                res_re2 = -np.dot(jvec, kern_sin * kern_part2 * kernel_im)
                res_im1 = np.dot(jvec, kern_cos * kern_part2 * kernel_re)

                res = (res_re1 + res_re2) + 1j*(res_im1 + res_im2)
            else:
                kernel = np.array([((2*np.exp(-w*db)*np.cos(w*dt) - (1 + np.exp(-2*w*db)))/w) for w in omega_grid])
                kern_re = [np.cos(w * self.dt*dkkp/2) * ( np.exp(-w*db*(k-kp-1)) + np.exp(-w*(self.b-db*(k-kp+1))) ) / (1-np.exp(-w*self.b)) for w in omega_grid]
                res_re = np.dot(jvec, kernel * kern_re)

                kern_im = [np.sin(w * self.dt*dkkp/2) * ( np.exp(-w*db*(k-kp-1)) - np.exp(-w*(self.b-db*(k-kp+1))) ) / (1-np.exp(-w*self.b)) for w in omega_grid]
                res_im = np.dot(jvec, kernel * kern_im)


                res = res_re + 1j*res_im

        return cut + res
