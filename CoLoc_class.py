import numpy as np
import pandas as pd
import itertools
import scipy as sp 

## general use functions 
    
def make_rca(q, binary = False):    
    """ 
    Computes the RCA index for a contingency table q 
    
    arguments: 
        q: pandas (pivoted) df 
        binary: whether to return raw RCA values (False) or a presencce-absence matrix by 'binarizing' the RCA matrix by setting every entry larger than one equal to 1, and lesser or equal to 1 to 0 (True).  Default is False
    
    returns: 
        RCA matrix (binary or raw)     
    """

    rca = (q.div(q.sum(axis=1), axis=0)).div((q.sum(axis=0)/q.sum().sum()),axis=1)

    # binarization
    if binary == True:
        rca = (rca > 1).astype(int)

    return rca

def make_EGmat(q):
    """
    Computes the Ellison-Glaeser co-agglomeration measure (Ellison et al., 2010) for a contingency table q. 
    
    arguments: 
        q: pandas (pivoted) df or 2d numpy array 
        
    returns: 
        EG co-agglomeration matrix
    """


    pc_i = q / q.sum(0)
    pc = q.sum(1) / q.sum().sum() 
    
    X = pc_i.sub(pc, axis = 0)
    
    return X.T.dot(X) / (1 - (pc**2).sum())

def make_PROXmat(q):
    """
    Computes the proximity measure (Hidalgo et al., 2007) for a contingency table q. 
    
    arguments: 
        q: pandas (pivoted) df
        
    returns: 
        proximity matrix
    """
    
    M = make_rca(q, binary = True)
    
    condprobs = M.T.dot(M) / M.sum(0)
    
    proxmat = np.minimum(condprobs, condprobs.T)
    
    proxmat -= np.diag(np.diag(proxmat))
    
    return  proxmat


def make_mockdata(shape = (50,10), n = 50*10*100):
    """
    Generates a contingency table with multinomially distributed counts with given shape. The parameter for the multinomial is constructed by generating a matrix containting random integers beween 1 and 100 and then normalizing the matrix.  
    
    arguments:
        shape: shape of matrix 
        n: total number of counts in matrix (number of trials in the multinomial) 
        
    returns: 
        contingency table (pandas dataframe) 
    """
    
    
    true_probs = np.random.randint(1,100, size = shape[0]*shape[1]) 
    true_probs = true_probs / true_probs.sum()
    
    counts = sp.stats.multinomial(n=n, p = true_probs).rvs().reshape(shape)  

    df = pd.DataFrame(counts)
    df = df.iloc[df.sum(1).argsort(),df.sum(0).argsort()]

    return df


def pseudocounts(q, prior = 'prop', nr_prior_obs = None):
    """"
    Computes the matrix of pseudocounts associated with a Dirichlet prior of given type for a contingency table q
    
    arguments: 
        q: pandas (pivoted) df or 2d numpy array 
        prior: The type of prior. Priors implemented: 
            'uniform': adds constant pseudocounts to each cell 
            'prop': add a pseudocount proportional to the product of the marginals in each cell (default)
            'none': sets pseudocounts to 0
        nr_prior_obs: the total number of counts in the resulting matrix of pesudocounts (i.e. the sum over all pesudocounts). This sets the weight of the prior. If nr_prior_obs is not given, the number of prior observations is taken to be equal to the total number of cells in the data (this way, the uniform prior adds a single observation to each cell.) Default is None
    
    returns: 
        matrix of pseudocounts 
    """

    if nr_prior_obs == None:
        nr_prior_obs = np.size(q)

    if prior == 'prop':
        alpha = nr_prior_obs*np.outer(q.sum(1),q.sum(0)) / (q.sum().sum())**2

    if prior == 'uniform':
        alpha = np.ones(np.shape(q))*nr_prior_obs/np.size(q)

    if prior == None:
        alpha = np.zeros(q.shape) 

        
    if isinstance(q, pd.DataFrame):
             alpha = pd.DataFrame(alpha, index = q.index, columns = q.columns)
        
    return alpha

    

#############################################################
# CoLoc class 

class CoLoc():
    """
    Class that enables computation of measures of location and co-location. Input argument is a contingency table q (thought of as having rows indexed by c and columns indexed by i), containing the number of observations in every cell (c,i). The rows c are thought of as 'locations' and the columns i as 'types' of economic activity. Counts thus typically represent number of workers in a given city and occupation, number of plants in a given region and industry or the number of dollar worth exported by a given country of a certain product. 
    
    arguments:
        q: pandas (pivoted) df or 2d numpy array 
        prior: The type of prior. Priors implemented: 
            'uniform': adds constant pseudocounts to each cell 
            'prop': add a pseudocount proportional to the product of the marginals in each cell (default)
            'none': sets pseudocounts to 0
        nr_prior_obs: the total number of counts in the resulting matrix of pesudocounts (i.e. the sum over all pesudocounts). This sets the weight of the prior. If nr_prior_obs is not given, the number of prior observations is taken to be equal to the total number of cells in the data (this way, the uniform prior adds a single observation to each cell.) Default is None
    
    A number of quantities are available as a property of the class:
        alpha: the matrix of pseudocounts representing the prior 
        qt: the matrix of observations and pseudocounts (q + alpha)
        qc: the number of counts per row c (qt summed over columns i) 
        qi: the number of counts per column i (qt summed over rows c) 
        q_tot: the total number of counts (qt summed over both columns and rows) 
        Ni: the number of columns i
        Nc: the number rows c 
        
        pci: the posterior mean of the probability of a random sample being of row c and column i 
        varpci: the posterior variance of the probability pci
        pc: the posterior mean of the probability of a random sample being of row c 
        pi: the posterior mean of the probability of a random sample being of column i
        
    After calling the function make_colocprobs() the following quantities are computed made available as property of the class: 
        
        pij: the posterior mean of the probability of two random samples from the same row are of column i and column j 
        varpij: the posterior variance of the probability pij
        pj_i: the posterior mean of the probability of a random sample being of column j given that a previous random sample from the same row was of column i 
        varpj_i: the posterior variance of the probability pj_i
    
        
    The following functions are available to compute each location measure (their posterior mean and standard deviation, respectively): 
        
        make_PMIpci(): returns matrix of location association estimates
        make_stdPMIpci(): returns the standard deviations of the posterior of the location associations
        make_sigPMIpci(): returns matrix of significantly nonzero location association estimates
        make_insigPMIpci(): returns matrix of insignificant locations association estimates (i.e. those that are not significantly nonzero)
        make_KLpc_i(): returns the estimated 'localization' of each column
        make_stdKLpc_i(): returns the standard deviation of the posterior of the 'localization' of each column 
        make_KLpi_c(): returns the estimated 'specialization' of each row
        make_stdKLpi_c(): returns the standard deviation of the posterior of the 'specialization' of each row
        make_MIpci(): returns the estimated 'overall specialization' 
        make_stdMIpci(): returns the standard deviation of the posterior of the 'overall specialization' 
        
    The following functions are available to compute each co-location measure (their posterior mean and standard deviation, respectively): 
        make_PMIpij: returns matrix of co-location association estimates for rows i 
        make_stdPMIpij(): returns standard deviations of posterior of the co-location associations
        make_sigPMIpij(): returns matrix of significantly nonzero co-location association estimates
        make_insig(PMIpij): returns matrix of insignificant co-location association estimates (i.e. those that are not significantly nonzero)
        make_KLpj_i(): returns the estimated 'co-dependence' of each column 
        make_stdKLpj_i(): returns the standard deviations of the posterior of the 'co-dependence' of each column 
        make_MIpij(): returns the estimated 'overall co-dependence' 
        
        NOTE: Computing any of the co-location measures automatically runs the function make_colocprobs() if is not run before. 
    """
    
    def __init__(self, q, prior = 'prop', nr_prior_obs = None):

        self.alpha = pseudocounts(q, prior=prior, nr_prior_obs = nr_prior_obs)
    
        
        self.qt = q + self.alpha 
        self.qc = self.qt.sum(1)
        self.qi = self.qt.sum(0)
        self.q_tot = self.qt.sum().sum() 
        
        self.colocprobs = False
        
        self.pci = self.qt/self.q_tot
        self.pc = self.qc / self.q_tot
        self.pi = self.qi / self.q_tot
        self.varpci = self.qt*(self.q_tot-self.qt)/(self.q_tot**2*(self.q_tot+1))
        
        self.pc_i = self.qt / self.qi 
        self.pi_c = (self.qt.T / self.qc).T 
        
        self.Ni = q.shape[1]
        self.Nc = q.shape[0]
    
    def make_PMIpci(self):
        """"
        Computes the location associations, given by the posterior mean of PMIci (i.e. the Bayesian estimate of PMIci).
        """
            
        PMI_hatpci = ((np.log(self.pci) - np.log(self.pi)).T - np.log(self.pc)).T
        
        term = (((-1/self.pci + 1/self.pi).T + 1/self.pc).T-1)/(2*(self.q_tot+1))
    
        return PMI_hatpci + term
            
    def make_stdPMIpci(self):
        """"
        Computes the standard deviation of the posterior of PMIci (i.e. the Bayesian uncertainty estimate of estimate of PMIci)
        """
        
                                                         
        return np.sqrt((((1/self.pci + 1/self.pi).T + 1/self.pc).T -3) / (self.q_tot+1))
    
    
    def make_sigPMIpci(self, epsilon = 0.05):
        """"
        Returns the location associations PMIpci, but sets estimates that are not significant by the probability of direction criterium equal to NaN. 
        
        epsilon is the threshold for significance (default is 0.05)
        """
        
        PMI = self.make_PMIpci()
        
        normprobs = sp.stats.norm.cdf(0, loc = np.abs(PMI), scale = self.make_stdPMIpci())
        sigPMI = PMI.where(normprobs<epsilon)
        
        return sigPMI
    
    def make_insigPMIpci(self, epsilon = 0.05):
        """"
        Returns the locations associations PMIpci, but sets estimates that are significant by the probability of direction criterium equal to NaN. 
        
        epsilon is the threshold for significance (default is 0.05)
        """
        
        PMI = self.make_PMIpci()
        
        normprobs = sp.stats.norm.cdf(0, loc = np.abs(PMI), scale = self.make_stdPMIpci())
        sigPMI = PMI.where(normprobs>epsilon)
        
        return sigPMI
        
    def make_KLpc_i(self):
        """"
        Computes the 'localization', given by hte posterior mean of KLpc_i (i.e. the Bayesian estimate of KLpc_i)
        """

        PMI_hatpci = (np.log(self.pc_i).T - np.log(self.pc)).T 

        KL_hat = (self.pc_i*PMI_hatpci).sum(0) 
                
        term =  (self.Nc - 1)/(2*(self.qi+1)) + ((self.pc_i.T/self.pc).sum(1) - 1 )/ (2*(self.q_tot+1))

        return KL_hat + term
        
    
    
    def make_stdKLpc_i(self):        
        """"
        Computes the standard deviation of the posterior of KLpc_i (i.e. the Bayesian uncertainty estimate of estimate of KLpc_i)
        """
        PMI_hatpci = (np.log(self.pc_i).T - np.log(self.pc)).T 

        KL_hat = (self.pc_i*PMI_hatpci).sum(0) 
        
        varKL = ((self.pc_i*PMI_hatpci**2).sum(0) - KL_hat**2)/(self.qi+1) + ((self.pc_i.T**2/self.pc).sum(1) - 1)/(self.q_tot+1) 

        return np.sqrt(varKL)

    
    def make_KLpi_c(self):
        """"
        Computes the 'specialization', given by the posterior mean of KLpi_c (i.e. the Bayesian estimate of KLpi_c)
        """

    
        PMI_hatpci = np.log(self.pi_c) - np.log(self.pi)

        KL_hat = (self.pi_c*PMI_hatpci).sum(1) 
                
        term =  (self.Ni - 1)/(2*(self.qc+1)) + ((self.pi_c/self.pi).sum(1) -1 )/ (2*(self.q_tot+1)) 

        return KL_hat + term
        
        

    def make_stdKLpi_c(self):
        """"
        Computes the standard deviation of the posterior of KLpi_c (i.e. the Bayesian uncertainty estimate of estimate of KLpi_c) 
        """

        PMI_hatpci = np.log(self.pi_c) - np.log(self.pi)

        KL_hat = (self.pi_c*PMI_hatpci).sum(1) 
        
        varKL = ((self.pi_c*PMI_hatpci**2).sum(1) - KL_hat**2)/(self.qc+1) + ((self.pi_c**2/self.pi).sum(1) - 1)/(self.q_tot+1) 
        
        return np.sqrt(varKL)
       
    def make_MIpci(self):
        """"
        Computes the 'overall specialization', given by the posterior mean of MIpci (i.e. the Bayesian estimate of MIpci)
        """
        
        PMI_hatpci = ((np.log(self.pci) - np.log(self.pi)).T - np.log(self.pc)).T

        MI_hat = (self.pci*PMI_hatpci).sum().sum()
        term = (self.qi.size-1)*(self.qc.size-1) / (2*(self.q_tot+1))

        return MI_hat + term 

    
    def make_stdMIpci(self):
        """"
        Computes the standard deviation of the posterior of MIpci (i.e. the Bayesian uncertainty estimate of MIpci)
        """
        PMI_hatpci = ((np.log(self.pci) - np.log(self.pi)).T - np.log(self.pc)).T

        MI_hat = (self.pci*PMI_hatpci).sum().sum()
        
        varMI = ((self.pci*PMI_hatpci**2).sum().sum() - MI_hat**2)/(self.q_tot+1)
        
        return np.sqrt(varMI)
        
        
    ##colocation functions    
    
    def make_colocprobs(self):
        """
        Computes the posterior mean and variance of the probabilities pij and pj_i, which are then available as properties of the class. 
        """
        
        self.pij = ((self.qt.T/(self.qc)).dot(self.qt))/self.q_tot 
    
        a = self.qt*(self.qt+1)
        b = self.qc*(self.qc+1) 
        first_term = (a.T/b).dot(a) 
        second_term = ((self.qt.T/self.qc)**2).dot(self.qt**2)
        self.varpij = (first_term - second_term)/(self.q_tot*(self.q_tot+1)) - self.pij**2/(self.q_tot+1)
    
    
        self.pj_i = (((self.qt.T/self.qc).dot(self.qt)).T / self.qi).T
        
        #varpj_i
        self.varpj_i = ((first_term - second_term)/(self.qi*(self.qi+1)) - (self.pj_i.T)**2/(self.qi+1)).T
            
        self.colocprobs = True
        

    def make_PMIpij(self):
        
        """"
        Computes the posterior mean of PMIij (i.e. the Bayesian estimate of PMIij) for each pairwise combination of columns of pivot table q
        """
        
        if not self.colocprobs:
            self.make_colocprobs()
        
    
        PMI_hatpij = (np.log(self.pij) - np.log(self.pi)).T - np.log(self.pi)
        
        term = (1/self.pi.values[:,None] + 1/self.pi.values[None,:]) / (2*(self.q_tot+1)) 
        
        return PMI_hatpij -self.varpij/(2*self.pij**2) + term 

    
    def make_stdPMIpij(self):
        """"
        Computes the standard deviation of the posterior of PMIij (i.e. the Bayesian uncertainty estimate of estimate of PMIij)
        """
        

        if not self.colocprobs:
            self.make_colocprobs()
        
          
        return np.sqrt(self.varpij/self.pij**2 +  (1/self.pi.values[:,None] + 1/self.pi.values[None,:] + 2*np.diag(1/self.pi) -4 ) / (self.q_tot+1) )
      
        
    
    def make_sigPMIpij(self, epsilon = 0.05):
        """"
        Returns the co-location associations PMIpij, but sets estimates that are not significant by the probability of direction criterium equal to NaN. 
        
        epsilon is the threshold for significance (default is 0.05)
        """
        
        PMI = self.make_PMIpij()
        
        normprobs = sp.stats.norm.cdf(0, loc = np.abs(PMI), scale = self.make_stdPMIpij())
        sigPMI = PMI.where(normprobs<epsilon)
        
        return sigPMI
    
    def make_insigPMIpij(self, epsilon = 0.05):
        """"
        Returns the co-locations associations PMIpij, but sets estimates that are significant by the probability of direction criterium equal to NaN. 
        
        epsilon is the threshold for significance (default is 0.05)
        """
      
        PMI = self.make_PMIpij()
        
        normprobs = sp.stats.norm.cdf(0, loc = np.abs(PMI), scale = self.make_stdPMIpij())
        sigPMI = PMI.where(normprobs>epsilon)
        
        return sigPMI
    
    
    def make_KLpj_i(self):           
        """"
        Computes the 'co-dependence', given by the posterior mean of KLpj_i (i.e. the Bayesian estimate of KLpj_i)
        """

        if not self.colocprobs:
            self.make_colocprobs()
        
        KL_hat = (self.pj_i*(np.log(self.pj_i)-np.log(self.pi))).sum(1) 
        
        term = (self.varpj_i / self.pj_i).sum(1)/2 + ((self.pj_i.T/self.pi).sum(1) -1 )/ (2*(self.q_tot+1)) 

        return KL_hat + term

    def make_stdKLpj_i(self):
        """"
        Computes the standard deviation of the posterior of KLpj_i (i.e. the Bayesian uncertainty estimate of estimate of KLpj_i) 
        """

        
        if not self.colocprobs:
            self.make_colocprobs()
            
        PMI_hatpj_i = np.log(self.pj_i.values) - np.log(self.pi.values)[None,:]              #[i.j]
        
        qt = self.qt.values
        qc = self.qc.values
        qi = self.qi.values
        pj_i = self.pj_i.values
        
        a = (qt*(qt+1))/(qc*(qc+1))[:,None] #[c,i]
        
        covterm1 = np.einsum('cj,ck,ci->ijk',qt,qt,a)    #[i,j,k]
        diagonal = a.T.dot(qt) #ij
        covterm1.reshape(self.Ni, self.Ni**2)[:,::self.Ni+1] += diagonal
        
        c = (qt/qc[:,None])**2   #[c,i]
        covterm2 = np.einsum('cj,ck,ci->ijk',qt,qt,c)   #[i,j,k]
        
        covterm3 = pj_i[:,None,:]*pj_i[:,:,None] #[i,j,k]

        covpj_ipk_i = (covterm1 - covterm2)/(qi*(qi+1))[:,None,None] - covterm3/(qi+1)[:,None,None]   #i,j,k
        
        varKLpj_i = np.einsum('ijk,ij,ik->i',covpj_ipk_i,PMI_hatpj_i,PMI_hatpj_i)
        
        varKLpj_i += ((pj_i**2/self.pi.values[None,:]).sum(1) - 1) / (self.q_tot+1)
    
        if isinstance(self.qt, pd.DataFrame):
             varKLpj_i = pd.Series(varKLpj_i, index = self.qt.columns)
            

        return np.sqrt(varKLpj_i)

        
    def make_MIpij(self):
        """"
        Computes the posterior mean of MIpij (i.e. the Bayesian estimate of MIpij)
        """
        

        if not self.colocprobs:
            self.make_colocprobs()
    
        PMI_hatpij = (np.log(self.pij) - np.log(self.pi)).T - np.log(self.pi) #[i,j]
        
        MI_hat = (self.pij*PMI_hatpij).sum().sum()
    
        return MI_hat + (self.varpij / self.pij).sum().sum()/2 - (self.Ni-1) / (self.q_tot+1)

    
    
#     def make_stdMIpij(self):
#        """"
#        Computes the standard deviation of the posterior of MIpij (i.e. the Bayesian uncertainty estimate of MIpij)
#        """
        
#         if not self.colocprobs:
#             self.make_colocprobs()
    
#         PMI_hatpij = (np.log(self.pij) - np.log(self.pi)).T - np.log(self.pi) #[i,j]
        
#         qt = np.array(self.qt)
#         qc = np.array(self.qc)
#         qi = np.array(self.qi)
#         pij = np.array(self.pij)
        
#         ####could go directly using einsum ('ci,cj,ck'), but need to deal with kronecker deltas... 
#         a = qt[:,None]*qt[:,:,None]

#         ni = qt.shape[1]
#         nc = qt.shape[0]
#         a.reshape(nc,ni**2)[:,::ni+1] += qt     #add q to diagonal when i = j

#         b = a[:,:,:,None]*qt[:,None,None,:]

#         b.reshape(nc,ni,ni**2)[:,:,::ni+1] += a #add a to diagonal when j=k
#         b.reshape(nc,ni**2,ni)[:,::ni+1,:] += a #add a to diagonal when i=k

#         ####figure out how to do without copying... 
#         c = b[:,:,:,:,None]*qt[:,None,None,None,:]  #[c,i,j,k,l]

#         c.reshape(nc,ni,ni,ni**2)[:,:,:,::ni+1] += b #add a to diagonal when k=l
#         c = c.swapaxes(2,3).copy()   #[c,i,k,j,l]
#         c.reshape(nc,ni,ni,ni**2)[:,:,:,::ni+1] += b #add a to diagonal when j=l
#         c = c.swapaxes(1,3).copy()   #[c,j,k,i,l]
#         c.reshape(nc,ni,ni,ni**2)[:,:,:,::ni+1] += b #add a to diagonal when i=l
#         c = c.swapaxes(0,4)/((qc+2)*(qc+3)) #[j,k,i,l,c]
#         c = np.transpose(c, axes = (4,2,0,1,3))   #[c,i,j,k,l]

#         covterm1 = c.sum(0)   #[i,j,k,l]
        
#         a = qt[:,None]*qt[:,:,None] #[c,i,j] 
#         a.reshape(qt.shape[0],qt.shape[1]**2)[:,::qt.shape[1]+1] += qt     #add q to diagonal when i = j 
#         a = (a.T/(qc+1)).T #[c,i,j]

#         covterm2 = np.einsum('cij,ckl->ijkl',a,a)
        
#         covterm3 = (pij[:,:,None,None]*pij[None,None,:,:])   #[i,j,k,l]

#         covpijpkl = (covterm1 - covterm2)/(self.q_tot*(self.q_tot+1)) - covterm3/(self.q_tot+1)
        
#         varMIpij = np.einsum('ijkl,ij,kl',covpijpkl,PMI_hatpij,PMI_hatpij)
            

#        return None 
    
    