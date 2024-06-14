from inflfn import IF_PT as influence_functional
from scipy import linalg
from scipy.linalg import svd as la_svd
import numpy as np
from time import time
!!!!!! from expbath_eta_kbc import eta
#import os
from opt_einsum import contract

class TEMPO_PT:

    def __init__ (self, bath: Bath, sp: SimulationParams):
        self.N = sp.N
        self.M = sp.M
        self.dt = sp.t_max/max(sp.N, 1)
        self.db = 1/(2*sp.M*sp.T)
        self.percentage = sp.cutoff

        self.h_0 = sp.H_S
        self.dh = sp.H_S.shape[0]
        self.dl = (self.dh)**2

        self.opA = sp.opA
        self.opB = sp.opB
        
        self.mps = []
        self.mpo = []
        self.maxdim = 0
        
        # basepath = Path(os.getcwd())
        # self.outpath = basepath / f'a={alpha},b={b},wc={wc}' / f't={t}' / f'M={M},N={N},r=1e-{r}'            
        
        self.eta = eta(bath, sp)
        
        # eta_t = np.zeros((self.N+1, 4), dtype=complex)
        # eta_ppmm_b = np.zeros((self.M, 2), dtype=complex)
        # eta_pmmp_b = np.zeros((2*self.M-2, 2), dtype=complex)
        # eta_pm_b_k = np.zeros(self.M, dtype=complex)
        # mix_eta_pp_arr = np.zeros((self.N+self.M, self.M), dtype=complex)
        # mix_eta_pm_arr = np.zeros((self.N+self.M, self.M), dtype=complex)
        
        time0 = time()

        eta_t, eta_ppmm_b, eta_pmmp_b, eta_pm_b_k, mix_eta_pp_arr, mix_eta_pm_arr = self.generate_eta_list()
        
        print('time for eta:', time() - time0)
    
        self.ifn = influence_functional(eta_t,
                                        eta_ppmm_b, eta_pmmp_b,
                                        eta_pm_b_k, mix_eta_pp_arr, mix_eta_pm_arr)

    def generate_eta_list (self):

        eta_t = np.zeros((self.N+1, 4), dtype=complex)
        eta_ppmm_b = np.zeros((self.M, 2), dtype=complex)
        eta_pmmp_b = np.zeros((2*self.M-2, 2), dtype=complex)
        eta_pm_b_k = np.zeros(self.M, dtype=complex)

        mix_eta_pp_arr = np.zeros((self.N+self.M, self.M), dtype=complex)
        mix_eta_pm_arr = np.zeros((self.N+self.M, self.M), dtype=complex)

        pp = self.eta.eta_pp_tt_k()
        pm = self.eta.eta_pm_tt_k()

        eta_t[0, :] = [pp, np.conj(pp), pm, np.conj(pm)]

        pp = self.eta.eta_pp_bb_k()
        eta_ppmm_b[0, :] = [pp, np.conj(pp)]

        ## k is assumed to be bigger than self.M
        for i in range(1,self.N+1):

            pp = self.eta.eta_pp_tt_kk(i)
            pm = self.eta.eta_pm_tt_kk(i)
            eta_t[i, :] = [pp, np.conj(pp), pm, np.conj(pm)]

            for j in range(0,self.M):
                mix_eta_pp_arr[i-1+self.M,j] = self.eta.eta_pp_mix_kk(i,j)
                mix_eta_pm_arr[i-1+self.M,j] = self.eta.eta_pm_mix_kk(i,j)
                
          
        eta_pmmp_b[0, :] = [0,0]
        eta_pm_b_k[0] = self.eta.eta_pm_bb_k(0)

        for i in range(1,2*self.M-2):
            pm = self.eta.eta_pm_bb_kk(i) #k+kp should run from 1 to 2M-3
            eta_pmmp_b[i, :] = [pm, np.conj(pm)]

        for i in range(1,self.M):
            pp = self.eta.eta_pp_bb_kk(i) # k-kp should run from 1 to M-1
            eta_ppmm_b[i, :] = [pp, np.conj(pp)]
            eta_pm_b_k[i] = self.eta.eta_pm_bb_k(i)

        return eta_t, eta_ppmm_b, eta_pmmp_b, eta_pm_b_k, mix_eta_pp_arr, mix_eta_pm_arr

    def get_mpo_block (self,k,kp):

        # Liouvillian dimension
        dl = self.dl
        
        # Input arguments assume that k > kp
        # Without memory truncation, expect the input to be k = n-1, and kp = current time (counting from 0) which is < n-1, where n is the total length of the MPS
        
        # In regular tempo, for propagating to t_k, this gets etas from eta_{k,k} to eta_{k,0}
        # In PT-TEMPO, at the kth timestep out of n timesteps, we would be getting etas from eta_{k,k} to eta_{n,k}
        #    If there is memory truncation, then from eta_{k,k} to eta_{kp,k} where kp > k
        
        ### the arrangement of indices of MPO is anticlockwise from 12 o'clock
        ###           < o
        ###          |    ^
        ###           > - |


        ### empty mpo list since every mpo chain at each round is new
        self.mpo = []

        if k == kp:
            self.mpo.append(self.ifn.I_k (k,k))
            return


        ### np.eye(n) is always a square matrix of size nxn
        delta = np.einsum('pa,ji->jpia',np.eye(dl),np.eye(dl))

        # leftmost
        b0 = np.einsum('ij,ja->jia',np.eye(dl),self.ifn.I_k (k,kp))
        
        # rightmost
        bend = np.einsum('ij,ja->jai',np.eye(dl),self.ifn.I_k (kp,kp))

        self.mpo.append(b0)


        ## gives [k-1,k-2,...,kp+1]
        for d in range(k-1, kp, -1):
            Ik = self.ifn.I_k (d, kp)
            Ik = np.einsum('jpia,ja->jaip',delta,Ik)
            #print('------IK-----',Ik.shape)
            self.mpo.append(Ik)

        self.mpo.append(bend)

    def initialize_mps(self, N):

        # Liouvillian dimension
        dl = self.dl
        
        ### the arrangement of indices of MPO is anticlockwise from 12 o'clock
        ###           < o
        ###          |    ^
        ###           > - |


        ### empty mpo list since every mpo chain at each round is new
        self.mps = []

        ###  NOTE THAT THIS ORDER IS BY THE DIAGRAM IN THE IPAD NOTES


        ### np.eye(n) is always a square matrix of size nxn
        delta = np.einsum('pa,ji->jpia',np.eye(dl),np.eye(dl))

        # leftmost
        b0 = self.ifn.I_k(N+self.M-1, 0)
        
        # rightmost
        bend = self.ifn.I_k(0, 0)

        self.mps.append(b0)


        ## gives [k-1,k-2,...,kp+1]
        for d in range(N+self.M-2, 0, -1):
            Ik = np.einsum('ab,ja->jab', np.eye(dl), self.ifn.I_k(d, 0))
            #print('------IK-----',Ik.shape)
            self.mps.append(Ik)

        self.mps.append(bend)

        # The mps now corresponds to the usual TEMPO tensors

        ll = len(self.mps)
                
        U, S, Vh = self.svd_truncate(self.mps[0], [0], [1], 0.0)
        SVh = S@Vh
        self.mps[0] = U
                
        for i in range(1, ll-2):
                #              |
                #    |---|   |---|
                #  a-|   |-b-|   |-c
                #    |---|   |---|
            temp = np.einsum('ab,jbc->jac', SVh, self.mps[i])
            U, S, Vh = self.svd_truncate(temp, [0,1], [2], 0.0)
            self.mps[i] = U
            SVh = S@Vh

        temp = np.einsum('ab,jbc->jac', SVh, self.mps[ll-2])
        self.mps[ll-2] = temp

        U, S, Vh = self.svd_truncate(self.mps[ll-1], [1], [0], 0.0)
        self.mps[ll-1] = np.transpose(Vh)
        US = U@S
        temp = np.einsum('jab,bc->jac', self.mps[ll-2], US)
        self.mps[ll-2] = temp
        # The mps now has orthogonality center at the second to last site (from the left)

    def get_correlator(self):

        # Hilbert space dimension
        dh = self.dh
        # Liouvillian dimension
        dl = self.dl

        t_corr = np.zeros(self.N, dtype=complex)
        
        # A operator is at the same location for both correlators
        # B operator is measured at mps[N+M-1] (between 0+ and 0-), as the equivalent of rho_0 in regular TEMPO
        # C operator is measured going from mps[N+M-2] to mps[N+M-3], only on the backwards branch (between 1- and 2-)

        # Real-time part of contour: from mps[0] to mps[N-1]
        # Imag-time part of contour: from mps[N] to mps[N+M-1]

        # Each measurement is broken up into 4 parts
        #  (leftmost/final times) @ (measurement at n*dt) @ (middle/initial times) @ (imag times)
        # 
        # First contract the imag indices (N, ..., N+M-1) with the system propagations and measurment at locations loc_sym and loc_therm
        # For each real time t at which another measurement is inserted,
        #   1) Compute the leftmost part and store in an array
        #   2) Grow the middle part and overwrite
        #   3) Contract everything; sym and therm will only differ in the portion used for imag times


        tt = time()
        
        expdb = linalg.expm(-self.db*self.h_0/2)
        expdt = linalg.expm(1j*self.dt*self.h_0/2)
        
        k3_t = (expdb@expdb).flatten()
        k3v_t = np.diag(k3_t * ([1.0+0.0j] * dl))
        th_imag = np.einsum('ii->i', k3v_t)

        # Measurements
        m_A = np.kron((self.opA@expdt).T, expdt.conj().T).T
        m_An = np.kron((expdt@self.opA@expdt).T, (expdt@expdt).conj().T).T

        # No measurements
        #   on the real time part of contour
        prop_re = np.kron((expdt@expdt).T, (expdt@expdt).conj().T).T
        
        print("prepare measurement components: ", time() - tt)

        # Need to duplicate each site index to describe the system-only time evolution in between
        # ordering indices such that it runs counterclockwise from the top left

        
        # sy_imag and th_imag constitute the rightmost part of the ensuing contraction
        
        # Propagate final
        prop_fin_arr = []
        endcap = np.zeros(dl, dtype=complex)
        for i in range(0, dh):
            endcap[i*(dh+1)] = complex(1)
        
        prop_fin_arr.append(endcap) # for t=t_f

        
        # eg, (iaj) = 
        #     i j
        #     | |
        #    |---|
        #    |   |-a
        #    |---|
        temp = contract('i,ia,ij->ja', prop_fin_arr[-1], self.mps[0], np.eye(dl, dtype=complex))
        prop_fin_arr.append(temp)

        for n in range(1, self.N-1):

            #     i j
            #     | |
            #    |---|
            #  a-|   |-b
            #    |---|
            temp = contract('ia,iab,ij->jb', prop_fin_arr[-1], self.mps[n], np.eye(dl, dtype=complex)) #check index of self.mps[n]
            prop_fin_arr.append(temp)
            
        print("build and store final propagations: ", time() - tt)
        
        tt = time()
        # Propagate real
        # This constitutes the middle part of the ensuing contraction and is updated at each step
        # This is the same for both sym and therm
        
        th_imag = self.mps[self.N-1]
        for n in range(1, self.N-1):
            temp = np.einsum('ia,ij->ja', prop_fin_arr[-n], m_An)
            t_corr[n-1] = np.einsum('ia,ia', temp, th_imag)
            ttt = time()
            temp = np.einsum('ij,ja->ia', prop_re, th_imag)
            th_imag = contract('iab,ij,jb->ia', self.mps[self.N - n - 1], np.eye(dl, dtype=complex), temp)
            print("step ", n, " update th_imag: ", time()-ttt)

        temp = np.einsum('ia,ij->ja', prop_fin_arr[1], m_An)
        t_corr[self.N-2] = np.einsum('ia,ia', temp, th_imag)

        temp = np.einsum('ij,ja->ia', prop_re, th_imag)
        th_imag = contract('ia,ij,ja->i', self.mps[0], np.eye(dl, dtype=complex), temp)    
        temp = np.einsum('i,ij->j', prop_fin_arr[0], m_A)
        t_corr[self.N-1] = np.einsum('i,i', temp, th_imag)
        
        print("propagate middle parts+contraction: ", time()-tt)
        
        return t_corr
    

    def make_pt(self):

        # Liouvillian dimension
        dl = self.dl
        
        print("initializing mps")
        self.initialize_mps(self.N)
        print("finished initializing mps")
        
        for n in range(1, self.N+self.M):

            tt = time()
            
            
            self.get_mpo_block (self.N+self.M-1, n)

            self.shift_ortho_center(len(self.mpo))

            self.zipup_mpo_mps_contraction()

            self.shift_ortho_center(0)
            self.shift_ortho_center(len(self.mps)-1)

            # if (n+1 == self.M + self.N): #(n+1 == self.M) or (n+1 == self.M + int(N / 2))
            #     np.savetxt(str(self.outpath / f'n={n}_bds.csv'), self.get_bond_dims(), delimiter=',')
            #     print("bond dims saved", self.get_bond_dims())

        print('maxdim:', self.maxdim)

        size = asizeof.asizeof(self.mps)
        print('size of self.mps',size)


    def make_thermpt(self):

        # Liouvillian dimension
        dl = self.dl

        expdb = linalg.expm(-self.db*self.h_0/2)
        expdt = linalg.expm(1j*self.dt*self.h_0/2)

        k3_t = (expdb@expdb).flatten()
        k3v_t = np.diag(k3_t * ([1.0+0.0j] * dl))
        k3_t = np.einsum('ii->i', k3v_t)
        prop_im = np.kron((expdb@expdb).T, expdb@expdb).T
        #m_C = np.kron((expdt@expdb).T, (expdb@self.opC@(expdt.conj().T))).T
        m_C = np.kron((expdt@expdb).T, (expdb@self.opB@(expdt.conj().T))).T

        print("initializing mps")
        self.initialize_mps(self.N)
        print("finished initializing mps")
        
        ## Here, I shift the ortho center to the site where the MPO-MPS contraction commences
        ## After MPO-MPS contraction, the MPS is right canonical
        ## I did a sweep from the right to the left and then another sweep from the left to the right
        ## other orders, like left to right and then right to left, make no big difference
        ## if the last MPS tensor is in imag time, I will absorb it into the left site.
        for n in range(1, self.N+self.M):

            tt = time()
            
            self.get_mpo_block (self.N+self.M-1, n)

            self.shift_ortho_center(len(self.mpo))

            self.zipup_mpo_mps_contraction()

            self.shift_ortho_center(0)
            self.shift_ortho_center(len(self.mps)-1)

            if n == 1:
                self.mps[-1] = contract('ia,ij,j->ia', self.mps[-1], np.eye(dl, dtype=complex), k3_t)
                self.mps[-2] = contract('iab,ij,jk,kb->ia', self.mps[-2], np.eye(dl, dtype=complex), prop_im, self.mps[-1])
                self.mps.pop()
            elif n < self.M:
                self.mps[-2] = contract('iab,ij,jk,kb->ia', self.mps[-2], np.eye(dl, dtype=complex), prop_im, self.mps[-1])
                self.mps.pop()

            elif n == self.M:
                self.mps[-1] = np.einsum('ij,ja->ia', m_C, self.mps[-1]) 
                self.mps[-2] = np.einsum('iab,ij,jb->ia',self.mps[-2], np.eye(dl, dtype=complex), self.mps[-1])
                self.mps.pop()


            # if (n+1 == self.M + self.N): #(n+1 == self.M) or (n+1 == self.M + int(N / 2))
            #     np.savetxt(str(self.outpath / f'n={n}_bds_contracted.csv'), self.get_bond_dims(), delimiter=',')
            #     print("bond dims saved", self.get_bond_dims())



    def make_sympt(self):

        # Liouvillian dimension
        dl = self.dl

        expdb = linalg.expm(-self.db*self.h_0/2)
        expdt = linalg.expm(1j*self.dt*self.h_0/2)

        k3_s = (expdb@self.opB@expdb).flatten()
        k3v_s = np.diag(k3_s * ([1.0+0.0j] * dl))
        k3_s = np.einsum('ii->i', k3v_s)
        connect = np.kron((expdb@expdt).T, (expdb@(expdt.conj().T))).T #same as m_C, with C = iden

        #k3_t = (expdb@expdb).flatten()
        #k3v_t = np.diag(k3_t * ([1.0+0.0j] * dl))
        #k3_t = np.einsum('ii->i', k3v_t)
        prop_im = np.kron((expdb@expdb).T, expdb@expdb).T
                
        print("initializing mps")
        self.initialize_mps(self.N)
        print("finished initializing mps")
        
        ## Here, I shift the ortho center to the site where the MPO-MPS contraction commences
        ## After MPO-MPS contraction, the MPS is right canonical
        ## I did a sweep from the right to the left and then another sweep from the left to the right
        ## other orders, like left to right and then right to left, make no big difference
        ## if the last MPS tensor is in imag time, I will absorb it into the left site.
        for n in range(1, self.N+self.M):

            tt = time()
            
            self.get_mpo_block (self.N+self.M-1, n)

            self.shift_ortho_center(len(self.mpo))

            self.zipup_mpo_mps_contraction()

            self.shift_ortho_center(0)
            self.shift_ortho_center(len(self.mps)-1)

            if n == 1:
                # self.mps[-1] = contract('ia,ij,j->ia', self.mps[-1], np.eye(dl, dtype=complex), k3_t)                
                self.mps[-1] = contract('ia,ij,j->ia', self.mps[-1], np.eye(dl, dtype=complex), k3_s)
                self.mps[-2] = contract('iab,ij,jk,kb->ia', self.mps[-2], np.eye(dl, dtype=complex), prop_im, self.mps[-1])
                self.mps.pop()
            elif n < self.M:
                self.mps[-2] = contract('iab,ij,jk,kb->ia', self.mps[-2], np.eye(dl, dtype=complex), prop_im, self.mps[-1])
                self.mps.pop()

            elif n == self.M:
                self.mps[-1] = np.einsum('ij,ja->ia', connect, self.mps[-1])
                self.mps[-2] = np.einsum('iab,ij,jb->ia',self.mps[-2], np.eye(dl, dtype=complex), self.mps[-1])
                self.mps.pop()


            # if (n+1 == self.M + self.N): #(n+1 == self.M) or (n+1 == self.M + int(N / 2))
            #     np.savetxt(str(self.outpath / f'n={n}_bds_contracted.csv'), self.get_bond_dims(), delimiter=',')
            #     print("bond dims saved", self.get_bond_dims())


    def get_therm0(self):

        # Liouvillian dimension
        dl = self.dl

        expdb = linalg.expm(-self.db*self.h_0/2)

        k3_t = (expdb@expdb).flatten()
        k3v_t = np.diag(k3_t * ([1.0+0.0j] * dl))
        k3_t = np.einsum('ii->i', k3v_t)
        prop_im = np.kron((expdb@expdb).T, expdb@expdb).T
        # m_C0 = np.kron((expdb).T, (expdb@self.opC@self.opA)).T #double-check order: CA or AC
        m_C0 = np.kron((expdb).T, (expdb@self.opB@self.opA)).T #double-check order: CA or AC
        
        print("initializing mps")
        self.initialize_mps(0)
        print("finished initializing mps")
        
        ## Here, I shift the ortho center to the site where the MPO-MPS contraction commences
        ## After MPO-MPS contraction, the MPS is right canonical
        ## I did a sweep from the right to the left and then another sweep from the left to the right
        ## other orders, like left to right and then right to left, make no big difference
        ## if the last MPS tensor is in imag time, I will absorb it into the left site.
        for n in range(1, self.M):
            
            self.get_mpo_block (self.M-1, n)

            self.shift_ortho_center(len(self.mpo))

            self.zipup_mpo_mps_contraction()

            self.shift_ortho_center(0)
            self.shift_ortho_center(len(self.mps)-1)

            if n == 1:               
                self.mps[-1] = contract('ia,ij,j->ia', self.mps[-1], np.eye(dl, dtype=complex), k3_t)
                self.mps[-2] = contract('iab,ij,jk,kb->ia', self.mps[-2], np.eye(dl, dtype=complex), prop_im, self.mps[-1])
                self.mps.pop()
            elif n < self.M-1:
                self.mps[-2] = contract('iab,ij,jk,kb->ia', self.mps[-2], np.eye(dl, dtype=complex), prop_im, self.mps[-1])
                self.mps.pop()
            elif n == self.M-1:
                self.mps[-2] = contract('ib,ij,jk,kb->i', self.mps[-2], np.eye(dl, dtype=complex), prop_im, self.mps[-1])
                self.mps.pop()
            
        self.mps[-1] = np.einsum('ij,j->i', m_C0, self.mps[-1])

        return self.mps[0][0] + self.mps[0][3]
    

    def get_sym0(self):

        # Liouvillian dimension
        dl = self.dl

        expdb = linalg.expm(-self.db*self.h_0/2)

        k3_s = (expdb@self.opB@expdb).flatten()
        k3v_s = np.diag(k3_s * ([1.0+0.0j] * dl))
        k3_s = np.einsum('ii->i', k3v_s)
        m_A0 = np.kron((self.opA@expdb).T, expdb).T

        #k3_t = (expdb@expdb).flatten()
        #k3v_t = np.diag(k3_t * ([1.0+0.0j] * dl))
        #k3_t = np.einsum('ii->i', k3v_t)
        prop_im = np.kron((expdb@expdb).T, expdb@expdb).T
        
        
        print("initializing mps")
        self.initialize_mps(0)
        print("finished initializing mps")
        
        ## Here, I shift the ortho center to the site where the MPO-MPS contraction commences
        ## After MPO-MPS contraction, the MPS is right canonical
        ## I did a sweep from the right to the left and then another sweep from the left to the right
        ## other orders, like left to right and then right to left, make no big difference
        ## if the last MPS tensor is in imag time, I will absorb it into the left site.
        for n in range(1, self.M):
            
            self.get_mpo_block (self.M-1, n)

            self.shift_ortho_center(len(self.mpo))

            self.zipup_mpo_mps_contraction()

            self.shift_ortho_center(0)
            self.shift_ortho_center(len(self.mps)-1)

            if n == 1:               
                self.mps[-1] = contract('ia,ij,j->ia', self.mps[-1], np.eye(dl, dtype=complex), k3_s)
                self.mps[-2] = contract('iab,ij,jk,kb->ia', self.mps[-2], np.eye(dl, dtype=complex), prop_im, self.mps[-1])
                self.mps.pop()
            elif n < self.M-1:
                self.mps[-2] = contract('iab,ij,jk,kb->ia', self.mps[-2], np.eye(dl, dtype=complex), prop_im, self.mps[-1])
                self.mps.pop()
            elif n == self.M-1:
                self.mps[-2] = contract('ib,ij,jk,kb->i', self.mps[-2], np.eye(dl, dtype=complex), prop_im, self.mps[-1])
                self.mps.pop()
            
        self.mps[-1] = np.einsum('ij,j->i', m_A0, self.mps[-1])

        return self.mps[0][0] + self.mps[0][3]
    

    def get_z(self):

        # Liouvillian dimension
        dl = self.dl

        expdb = linalg.expm(-self.db*self.h_0/2)

        k3_s = (expdb@expdb).flatten()
        k3v_s = np.diag(k3_s * ([1.0+0.0j] * dl))
        k3_s = np.einsum('ii->i', k3v_s)
        m_A0 = np.kron((expdb).T, expdb).T

        #k3_t = (expdb@expdb).flatten()
        #k3v_t = np.diag(k3_t * ([1.0+0.0j] * dl))
        #k3_t = np.einsum('ii->i', k3v_t)
        prop_im = np.kron((expdb@expdb).T, expdb@expdb).T
        
        
        print("initializing mps")
        self.initialize_mps(0)
        print("finished initializing mps")
        
        ## Here, I shift the ortho center to the site where the MPO-MPS contraction commences
        ## After MPO-MPS contraction, the MPS is right canonical
        ## I did a sweep from the right to the left and then another sweep from the left to the right
        ## other orders, like left to right and then right to left, make no big difference
        ## if the last MPS tensor is in imag time, I will absorb it into the left site.
        for n in range(1, self.M):
            
            self.get_mpo_block (self.M-1, n)

            self.shift_ortho_center(len(self.mpo))

            self.zipup_mpo_mps_contraction()

            self.shift_ortho_center(0)
            self.shift_ortho_center(len(self.mps)-1)

            if n == 1:               
                self.mps[-1] = contract('ia,ij,j->ia', self.mps[-1], np.eye(dl, dtype=complex), k3_s)
                self.mps[-2] = contract('iab,ij,jk,kb->ia', self.mps[-2], np.eye(dl, dtype=complex), prop_im, self.mps[-1])
                self.mps.pop()
            elif n < self.M-1:
                self.mps[-2] = contract('iab,ij,jk,kb->ia', self.mps[-2], np.eye(dl, dtype=complex), prop_im, self.mps[-1])
                self.mps.pop()
            elif n == self.M-1:
                self.mps[-2] = contract('ib,ij,jk,kb->i', self.mps[-2], np.eye(dl, dtype=complex), prop_im, self.mps[-1])
                self.mps.pop()
            
        self.mps[-1] = np.einsum('ij,j->i', m_A0, self.mps[-1])

        return self.mps[0][0] + self.mps[0][3]
