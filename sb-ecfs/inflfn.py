import numpy as np


## the physical index is first then the internal bond
class IF_CTC:
                      
    def __init__(self, eta_k: np.ndarray, eta_kk: np.ndarray):

        self.eta_k = eta_k
        self.eta_kk = eta_kk
        self.N = (len(self.eta_k) - 2) // 2

    def I_k(self,k,kp):

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
            sarray = np.array([1,1])

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
            sarray = np.array([[1,-1],[-1,1]])
            return np.array([[np.exp(-eta*1*1),np.exp(-eta*1*-1)],\
                             [np.exp(-eta*1*-1),np.exp(-eta*-1*-1)]],dtype=np.complex128)

class IF_PT:

    def __init__ (self, eta_t: np.ndarray,
                        eta_ppmm_b: np.ndarray, eta_pmmp_b: np.ndarray, eta_pm_b_k: np.ndarray,
                        eta_mix_pp_arr: np.ndarray, eta_mix_pm_arr: np.ndarray):

        self.N = size(eta_t)[0] - 1
        self.M = size(eta_ppmm_b)[0]
        
        # The following are numpy arrays of the specified shapes:
        self.eta_t = eta_t # (self.N+1, 4)
        self.eta_ppmm_b = eta_ppmm_b # (self.M, 2)
        self.eta_pmmp_b = eta_pmmp_b # (2*self.M-2, 2)
        self.eta_pm_b_k = eta_pm_b_k # (self.M, )
        self.eta_mix_pp_arr = eta_mix_pp_arr # (self.N+self.M, self.M)
        self.eta_mix_pm_arr = eta_mix_pm_arr # (self.N+self.M, self.M)


    ## ++, --, +-, -+
    def phi_k(self, fkp: int, fkpp: int, fkm: int, fkpm: int,
                    eta_k_arr: ndarray):

        return -(eta_k_arr[0]*fkp*fkpp + eta_k_arr[1]*fkm*fkpm + eta_k_arr[2]*fkp*fkpm + eta_k_arr[3]*fkm*fkpp)



    def I_k (self, k: int, kp: int):
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
