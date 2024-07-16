import numpy as np

class IF_CTC:
    '''
    Class defining the influence functional on the complex-time contour

    Attributes
    ----------
    N: int
        Number of timesteps to take along one leg of the contour
    eta_k: ndarray
        Array of eta values for time-local contributions to influence functional
    eta_kk: ndarray
        Array of eta values for time-nonlocal contributions to the influence functional
    '''
    def __init__(self, eta_k: np.ndarray, eta_kk: np.ndarray):

        self.eta_k = eta_k
        self.eta_kk = eta_kk
        self.N = (len(self.eta_k) - 2) // 2

    def I_k(self, k: int, kp: int):
        '''
        Computes the nonzero-elements of the b-tensor describing temporal correlations between two contour times

        Parameters
        ----------
        k: int
            Index of the time along the complex-time contour ("contour time")
        kp: int
            Index of the time along the complex-time contour ("contour time"), assumed to be less than/comes before `k`

        Returns
        -------
        res: ndarray
            Non-zero elements of the b-tensor describing temporal correlations between contour times `k` and `kp`        
        '''
        ## if k < kp, swap them
        if k < kp:
            kdum = k
            k = kp
            kp = kdum

        if kp == 0:

            return np.array([[1,1],[1,1]],dtype=np.complex128)

        if k == 2*self.N+1:

            return np.array([[1,1],[1,1]],dtype=np.complex128)

        if k == kp:
            eta = self.eta_k[k]

            return np.array([[[np.exp(-eta*1),0],\
                        [0,0]],\
                        \
                        [[0,0],\
                        [0,np.exp(-eta*1)]]],dtype=np.complex128)
            
        
        elif k != kp:
            if k > kp:
                eta = self.eta_kk[k, kp]
            else:
                eta = self.eta_kk[kp, k]

            return np.array([[np.exp(-eta*1*1),np.exp(-eta*1*-1)],\
                             [np.exp(-eta*1*-1),np.exp(-eta*-1*-1)]],dtype=np.complex128)

class IF_PT:
    '''
    Class defining the influence functional on the Kadanoff-Baym-like contour

    Attributes
    ----------
    N: int
        Number of timesteps along the real-time part of the contour
    M: int
        Number of timesteps along the half of the imaginary-time part of the contour
    eta_t: ndarray
        Array of eta values describing real-time, real-time correlations in the influence functional
    eta_ppmm_b: ndarray
        Matrix of eta values describing imaginary-time, imaginary-time correlations in the influence functional, considering only the plus branches of both contour times
    eta_pmmp_b: ndarray
        Matrix of eta values describing imaginary-time, imaginary-time correlations in the influence functional, considering the plus branch of one contour time and the minus branch for the other
    eta_pm_b_k: ndarray
        Matrix of eta values describing the time-local, imaginary-time, imaginary-time correlations in the influence functional, considering the plus branch of a contour time and the minus branch of the same contour time
    eta_mix_pp_arr: ndarray
        Matrix of eta values describing the real-time, imaginary-time correlations in the influence functional, taking the plus branch of the contour for both real- and imaginary-times
    eta_mix_pm_arr: ndarray
        Matrix of eta values describing the real-time, imaginary-time correlations in the influence functional, taking the plus branch of the contour for real-time and the minus branch of the contour for imaginary-time
    '''
    def __init__ (self, eta_t: np.ndarray,
                        eta_ppmm_b: np.ndarray, eta_pmmp_b: np.ndarray, eta_pm_b_k: np.ndarray,
                        eta_mix_pp_arr: np.ndarray, eta_mix_pm_arr: np.ndarray):

        self.N = eta_t.shape[0] - 1
        self.M = eta_ppmm_b.shape[0]
        
        # The following are numpy arrays of the specified shapes:
        self.eta_t = eta_t # (self.N+1, 4)
        self.eta_ppmm_b = eta_ppmm_b # (self.M, 2)
        self.eta_pmmp_b = eta_pmmp_b # (2*self.M-2, 2)
        self.eta_pm_b_k = eta_pm_b_k # (self.M, )
        self.eta_mix_pp_arr = eta_mix_pp_arr # (self.N+self.M, self.M)
        self.eta_mix_pm_arr = eta_mix_pm_arr # (self.N+self.M, self.M)


    def phi_k(self, fkp: int, fkpp: int, fkm: int, fkpm: int,
                    eta_k_arr: np.ndarray) -> complex:
        '''
        Computes the exponent of the portion of the influence functional described by a single b-tensor, for the given values of the system states at two contour times, k and kp.

        Parameters
        ----------
        fkp: int
            System state at contour time `k`, on the plus branch of the contour
        fkpp: int
            System state at contour time `kp`, on the plus branch of the contour
        fkm: int
            System state at contour time `k`, on the minus branch of the contour
        fkpm: int
            System state at contour time `kp`, on the minus branch of the contour
        eta_k_arr: ndarray
            Array of eta values correlating between different branches of the contour at the contour times `k` and `kp`. Ordering of the entries is assumed to be (++, --, +-, -+), where each pair corresponds to the pair of contour times (`k`, `kp`)

        Returns
        -------
        res: complex
            Value of the exponent of the portion of the influence functional described by a single b-tensor
        '''
        return -(eta_k_arr[0]*fkp*fkpp + eta_k_arr[1]*fkm*fkpm + eta_k_arr[2]*fkp*fkpm + eta_k_arr[3]*fkm*fkpp)


    def I_k (self, k: int, kp: int) -> np.ndarray:
        '''
        Computes the nonzero-elements of the b-tensor describing temporal correlations between two contour times

        Parameters
        ----------
        k: int
            Index of the time along the Kadanoff-Baym-like contour ("contour time")
        kp: int
            Index of the time along the Kadanoff-Baym-like contour ("contour time"), assumed to be less than/comes before `k`

        Returns
        -------
        res: ndarray
            Non-zero elements of the b-tensor describing temporal correlations between contour times `k` and `kp`
        '''
        ## switch if k < kp
        if k < kp:
            kdum = k
            k = kp
            kp = kdum

        if k <= self.M-1 and kp <= self.M-1:
            if k == kp:
                eta = [self.eta_ppmm_b[0, 0],
                       self.eta_ppmm_b[0, 1],
                       self.eta_pm_b_k[k],
                       np.conjugate(self.eta_pm_b_k[k])]
                
            else:
                eta = [self.eta_ppmm_b[k-kp, 0],
                       self.eta_ppmm_b[k-kp, 1],
                       self.eta_pmmp_b[k+kp, 0],
                       self.eta_pmmp_b[k+kp, 1]]
                
        elif k >= self.M and kp >= self.M:
            eta = self.eta_t[k-kp]
            

        else:
            eta = [self.eta_mix_pp_arr[k,kp],
                   np.conjugate(self.eta_mix_pp_arr[k,kp]),
                   self.eta_mix_pm_arr[k,kp],
                   np.conjugate(self.eta_mix_pm_arr[k,kp])]
            

        r = np.array([[np.exp(self.phi_k(1,1,1,1,eta)),   np.exp(self.phi_k(1,1,1,-1,eta)),   np.exp(self.phi_k(1,-1,1,1,eta)),   np.exp(self.phi_k(1,-1,1,-1,eta))], \
                      [np.exp(self.phi_k(1,1,-1,1,eta)),  np.exp(self.phi_k(1,1,-1,-1,eta)),  np.exp(self.phi_k(1,-1,-1,1,eta)),  np.exp(self.phi_k(1,-1,-1,-1,eta))], \
                      [np.exp(self.phi_k(-1,1,1,1,eta)),  np.exp(self.phi_k(-1,1,1,-1,eta)),  np.exp(self.phi_k(-1,-1,1,1,eta)),  np.exp(self.phi_k(-1,-1,1,-1,eta))],\
                      [np.exp(self.phi_k(-1,1,-1,1,eta)), np.exp(self.phi_k(-1,1,-1,-1,eta)), np.exp(self.phi_k(-1,-1,-1,1,eta)), np.exp(self.phi_k(-1,-1,-1,-1,eta))]])


        if k == 0:

            a = np.diag([np.exp(self.phi_k(1,1,1,1,eta)),
                         np.exp(self.phi_k(1,1,-1,-1,eta)),
                         np.exp(self.phi_k(-1,-1,1,1,eta)),
                         np.exp(self.phi_k(-1,-1,-1,-1,eta))])

            return a

        elif k == kp:
            b = np.diag([np.exp(self.phi_k(1,1,1,1,eta)),
                         np.exp(self.phi_k(1,1,-1,-1,eta)),
                         np.exp(self.phi_k(-1,-1,1,1,eta)),
                         np.exp(self.phi_k(-1,-1,-1,-1,eta))])
            
            return b

        else:
            
            return r
