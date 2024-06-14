from inflfn import IF_CTC as influence_functional
!!!!!! from expbath_eta_ctc import eta

class TEMPO_CTC():
    
    def __init__(self, bath: Bath, sp: SimulationParams):
    
        self.N = sp.N
        self.M = sp.M
        self.dt = (sp.t_max - 1j/(2*sp.T))/sp.N
        self.db = 1/(2*sp.M*sp.T)
        self.percentage = 10**(-r)

        self.h_0 = sp.H_S
        self.dh = sp.H_S.shape[0]
        self.dl = (self.dh)**2

        self.opA = sp.opA
        self.opB = sp.opB

        self.percentage = sp.cutoff
        self.maxdim = sp.maxdim
        
        self.eta = eta_Bose(bath, sp)
        self.eta_k = np.zeros(2*self.N + 2, dtype=complex)
        self.eta_kk = np.zeros((2*self.N+2, 2*self.N+2), dtype=complex)

        ek = self.eta.Eta_k(0)
        for i in range(0, 2*self.N+1):
            for j in range(1, i):
                self.eta_kk[i, j] = self.eta.eta_kk(i,j)

        for i in range(0, self.N+1):
            self.eta_k[i] = ek
        for i in range(self.N+1, 2*self.N+1):
            self.eta_k[i] = np.conj(ek)
        ## all physical indices of the MPS 2N+1 - --- - 0
        self.ind_arr = np.arange(2*self.N+1,-1,-1)
        
        self.ifn = influence_functional(self.eta_k, self.eta_kk)
        
        self.maxbonddim_both = []
        self.maxbonddim_sweep = []


    ## this MPS is made to be left canonical
    def get_MPS(self, opB):

        self.mps = []

        Uprop = expm(-1j*self.h_0*self.dt/2).T
        U, S, Vh = np.linalg.svd(Uprop, full_matrices=False)
        SVh = np.einsum('a,as->sa',S,Vh)

        self.mps.append(U)

        for _ in range(self.N-1):
            Uprop = expm(-1j*self.h_0*self.dt).T
            U, S, Vh = np.linalg.svd(Uprop, full_matrices=False)

            ## SVh is from previous step
            self.mps.append(np.einsum('sa,sb->sab',SVh,U))

            ## update SVh
            SVh = np.einsum('a,as->sa',S,Vh)

        # Uprop = (expm(1j*self.h_0*self.dt.conjugate()/2)@self.opB\
        #         @expm(-1j*self.h_0*self.dt/2)).T
        Uprop = (expm(1j*self.h_0*self.dt.conjugate()/2)@opB\
                @expm(-1j*self.h_0*self.dt/2)).T
        U, S, Vh = np.linalg.svd(Uprop, full_matrices=False)
        self.mps.append(np.einsum('sa,sb->sab',SVh,U))
        SVh = np.einsum('a,as->sa',S,Vh)

        for _ in range(self.N-1):
            Uprop = expm(1j*self.h_0*self.dt.conjugate()).T
            U, S, Vh = np.linalg.svd(Uprop, full_matrices=False)

            ## SVh is from previous step
            self.mps.append(np.einsum('sa,sb->sab',SVh,U))

            ## update SVh
            SVh = np.einsum('a,as->sa',S,Vh)

        Uprop = expm(1j*self.h_0*self.dt.conjugate()/2).T
        U, S, Vh = np.linalg.svd(Uprop, full_matrices=False)

        ## SVh is from previous step
        self.mps.append(np.einsum('sa,sb->sab',SVh,U))

        ## update SVh
        SVh = np.einsum('a,as->sa',S,Vh)

        self.mps.append(SVh)
        
        # Left-canonicalize MPS for MPO-MPS contraction later
        ll = len(self.mps)
        U, S, Vh = self.svd_truncate(self.mps[0], [0], [1], 0.0)
        SVh = S@Vh
        self.mps[0] = U
                
        for i in range(1, ll-1):
            temp = np.einsum('ab,jbc->jac', SVh, self.mps[i])
            U, S, Vh = self.svd_truncate(temp, [0,1], [2], 0.0)
            self.mps[i] = U
            SVh = S@Vh
        temp = np.einsum('ab,jb->ja', SVh, self.mps[ll-1])
        self.mps[ll-1] = temp



    def get_MPO(self,ind_arr,ind):
        self.mpo = []

        delta = np.einsum('pa,ji->jpia',np.eye(2),np.eye(2))

        ## prep for tensors to add to F_MPO_k list
        # print('ind_arr[0]=',ind_arr[0])
        # print('ind=',ind)
        # print('I_k',self.ifn.I_k (ind_arr[0],ind))
        F_1 = np.einsum('sz,sb->szb',np.eye(2),self.ifn.I_k (ind_arr[0],ind))
        self.mpo.append(F_1)

        for i in range(1,len(ind_arr)-1):
            if ind_arr[i] != ind:
                self.mpo.append(np.einsum('jpia,jp->jpia',delta,self.ifn.I_k (ind_arr[i],ind)))
            else:
                self.mpo.append(np.einsum('ji,ajb->iajb',np.eye(2),self.ifn.I_k (ind_arr[i],ind)))
            #print('newly added mpo',np.einsum('jpia,jp->jpia',delta,self.ifn.I_k (ind_arr[i],ind)))
            #sys.quit()
        #print('I_k',self.ifn.I_k (ind_arr[-1],ind))
        F_end = np.einsum('sz,sb->sbz',np.eye(2),self.ifn.I_k (ind_arr[-1],ind))
        self.mpo.append(F_end)

    def decimate_site(self, si):
        # First shift orthogonality center to site si
        # Sum over site index, subsume result into site si-1
        # Left canonicalize final result
        
        ll = len(self.mps)
        d = self.mps[0].shape[0]
        triv = np.ones(d)

        U, S, Vh = self.svd_truncate(self.mps[ll-1], [1], [0], 0.0)

        self.mps[ll-1] = Vh.T
        US = U@S
        
        for i in range(ll-2, si, -1):
            temp = np.einsum('iab,bc->aic', self.mps[i], US)
            U, S, Vh = self.svd_truncate(temp, [0], [1,2], 0.0)
            US = U@S
            self.mps[i] = np.moveaxis(Vh, [0, 1, 2], np.argsort([1, 0, 2]))
            
        #for i in range(0,d):
        #    print((self.mps[si])[i, :, :])
        #    print("\n")
        temp = np.einsum('i,iab,bc->ac', triv, self.mps[si], US)
        #self.mps[si] = np.einsum('iab,bc->iac', self.mps[si], US)
        #temp = np.sum(self.mps[si],axis=0)
        #print(mytemp)
        #print("\n\n")
        #print(temp)
        #print("=====\n=====\n")
        if si+1 == ll-1:
            self.mps[si+1] = np.einsum('ab,ib->ia', temp, self.mps[si+1])
        elif si+1 < ll-1:
            self.mps[si+1] = np.einsum('ab,ibc->iac', temp, self.mps[si+1])
        
            U, S, Vh = self.svd_truncate(self.mps[si+1], [0,1], [2], 0.0)
            self.mps[si+1] = U
            SVh = S@Vh
            for i in range(si+2, ll-1):
                temp = np.einsum('ab,jbc->jac', SVh, self.mps[i])
                U, S, Vh = self.svd_truncate(temp, [0,1], [2], 0.0)
                self.mps[i] = U
                SVh = S@Vh
            temp = np.einsum('ab,jb->ja', SVh, self.mps[ll-1])
            self.mps[ll-1] = temp
        self.mps.pop(si)

    def propagate(self, opA, opB):

        print('canonicalized zipup; canonical decimation until the last 4')

        self.ind_arr = np.arange(2*self.N+1,-1,-1)
        self.maxbonddim_both = []
        self.maxbonddim_sweep = []
        
        #self.get_MPS()
        self.get_MPS(opB)

        print('self.dt=',self.dt)
        
        contract_time = 0.0
        mpo_time = 0.0
        decim_time = 0.0

        while len(self.ind_arr) > 4:

            i = len(self.ind_arr)//2
            tt = time()
            self.get_MPO(self.ind_arr,self.ind_arr[i])
            mpo_time += (time() - tt)

            tt = time()
            self.zipup_mpo_mps_contraction()
            contract_time += (time() - tt)

            tt = time()
            self.decimate_site(i)
                
            self.ind_arr = np.delete(self.ind_arr,i)
            decim_time += (time() - tt)

        ## OUTSIDE OF THE WHILE LOOP
        # 
        # This is when mps has 4 elements
        # so the tensors that receive the summed over tensors
        # have different shapes 
            
        i = len(self.ind_arr)//2
        print('---i---',i)
        tt = time()
        self.get_MPO(self.ind_arr,self.ind_arr[i])
        mpo_time += (time() - tt)

        tt = time()
        self.zipup_mpo_mps_contraction()
        contract_time += (time() - tt)
        
        tt = time()
        self.mps[i] = np.sum(self.mps[i],axis=0)
        self.mps[i+1] = np.einsum('jk,ik->ij',self.mps[i],self.mps[i+1])
        self.mps.pop(i)
        self.ind_arr = np.delete(self.ind_arr,i)
        decim_time += (time() - tt)


        ## this is when mps has 3 elements
        i = len(self.ind_arr)//2
        tt = time()
        self.get_MPO(self.ind_arr,self.ind_arr[i])
        mpo_time += (time() - tt)
        
        tt = time()
        self.zipup_mpo_mps_contraction()
        contract_time += (time() - tt)

        tt = time()
        self.mps[i] = np.sum(self.mps[i],axis=0)
        #print('shape of mps[i-1]',self.mps[i-1].shape)
        #print('shape of mps[i]',self.mps[i].shape)
        self.mps[i-1] = np.einsum('ik,kl->il',self.mps[i-1],self.mps[i])
        self.mps.pop(i)
        self.ind_arr = np.delete(self.ind_arr,i)
        decim_time += (time() - tt)

        ## check the dimension
        ## this is when mps has 2 elements
        ## after einsum, dimension is (s_2N+1,s_0)
        #print('before corr',np.einsum('ij,kj->ik',self.mps[0],self.mps[1]).T)
        

        tt = time()
        # correlator = np.einsum('ij,kj->ik',self.mps[0],self.mps[1])@self.opA
        correlator = np.einsum('ij,kj->ik',self.mps[0],self.mps[1])@opA
        print("final contraction: ", time()-tt)

        #print('correlator',correlator)
        ## take the trace
        tt = time()
        correlator = np.trace(correlator)
        print("final trace: ", time()-tt)

        #print(self.maxbonddim)
        
        print("time to do contractions: ", contract_time)
        print("time to make mpo: ", mpo_time)
        print("time to decimate site: ", decim_time)

        #print(self.maxbonddim)
        
        return correlator, np.average(np.array(self.maxbonddim_both)), np.average(np.array(self.maxbonddim_sweep))
