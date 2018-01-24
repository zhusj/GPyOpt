# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .base import AcquisitionBase
from ..util.general import get_quantiles
import numpy as np

import matplotlib.pyplot as plt

class AcquisitionGES(AcquisitionBase):
    """
    Greedy entropy search acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function
    :param jitter: positive value to make the acquisition more explorative.

    .. Note:: allows to compute the Improvement per unit of cost

    """

    analytical_gradient_prediction = True

    def __init__(self, model, space, optimizer=None, cost_withGradients=None, jitter=0.01):
        self.optimizer = optimizer
        super(AcquisitionGES, self).__init__(model, space, optimizer, cost_withGradients=cost_withGradients)
        self.jitter = jitter
        
    @staticmethod
    def fromConfig(model, space, optimizer, cost_withGradients, config):
        return AcquisitionGES(model, space, optimizer, cost_withGradients, jitter=config['jitter'])

    def _compute_acq(self, x):
        """
        Computes the Greedy entropy search per unit of cost
        """

        # print "x,shape: ", x.shape
        if x.shape[0]>1:
            # x = x[:1000,:]  
            # print "x: ", x
            m, V = self.model.predict(x, full_cov=True)

            # if isinstance(V, np.ndarray):
            #     V[V<1e-10] = 1e-10
            # elif V< 1e-10:
            #     V = 1e-10
            # fmin = self.model.get_fmin()
            number = 10000
            print("number: ", number)          
            self.pmin = self.joint_pmin(m, V, number)
            self.logP = np.log(self.pmin)
            # eps = 1e-9
            # H = -np.multiply(self.pmin, (self.logP+eps))
            H = -np.multiply(self.pmin, (self.logP))
      
            f_acqu = np.array([H])
            f_acqu = f_acqu.T
            # print "f_acqu: ", f_acqu



            # m, v = self.model.predict(x)
            # print "v: ", v
            # raw_input("press to continue...")

            # f_acqu = v







            self.x = x
            self.f_acqu = f_acqu

            self.max_f_acqu = np.max(f_acqu)
            
                # print "f_acqu.shape: ", f_acqu  

            # else:
            #     print "x: ", x            

            #     if np.isnan(x[0][0]):
            #         f_acqu = self. lasf_f_acqu
            #     else:
            #         i,j = np.where(self.x==x[0][0])
            #         print "i,j: ", i,j
            #         f_acqu = self.f_acqu[i,j]
            #         f_acqu = np.array([f_acqu,])
            #         # print "f_acqu: ", f_acqu
            #         self. lasf_f_acqu = f_acqu
                
            # print "f_acqu.shape: ", f_acqu.shape   

            # plt.plot(x)
            # raw_input("Press Enter to continue...")
            # print "self.pmin:", self.pmin
            # print "f_acqu: ", f_acqu
            # print "self.max_f_acqu: ", self.max_f_acqu

            return f_acqu
        else:
            return np.array([self.max_f_acqu])
  
    # def _compute_acq_withGradients(self, x):
    #     """
    #     Computes the Greedy entropy search and its derivative (has a very easy derivative!)
    #     """
    #     f_acqu = self._compute_acq(x)

        
    #     D = x.shape[1]
    #     e = 1.0e-5
    #     ddHdx = np.zeros((1, D))
    #     for d in range(D):
    #         # ## First part:
    #         y = np.array(x)
    #         y[0, d] += e
    #         dHy1 = self._compute_acq(x)
    #         # ## Second part:
    #         y = np.array(x)
    #         y[0, d] -= e
    #         dHy2 = self._compute_acq(x)

    #         ddHdx[0, d] = np.divide((dHy1 - dHy2), 2 * e)
    #         ddHdx = -ddHdx
    #     # endfor
    #     if len(ddHdx.shape) == 3:
    #         df_acqu = ddHdx
    #     else:
    #         df_acqu = np.array([ddHdx])

    #     return f_acqu, df_acqu

    def _compute_acq_withGradients(self, x):
        """
        Computes the Greedy entropy search and its derivative (has a very easy derivative!)
        """
        f_acqu = self._compute_acq(x)

        fmin = self.model.get_fmin()
        m, s, dmdx, dsdx = self.model.predict_withGradients(x)
        phi, Phi, _ = get_quantiles(self.jitter, fmin, m, s)    
        df_acqu = dsdx * phi - Phi * dmdx
        return f_acqu, df_acqu

    def innovations(self, x, rep):
        # Get the variance at x with noise
        _, v = self.model.predict(x)

        # Get the variance at x without noise
        v_ = v - self.sn2

        # Compute the variance between the test point x and
        # the representer points
        sigma_x_rep = self.model.predict_variance(rep, x)

        norm_cov = np.dot(sigma_x_rep, np.linalg.inv(v_))
        # Compute the stochastic innovation for the mean
        dm_rep = np.dot(norm_cov, np.linalg.cholesky(v + 1e-10))
        dm_rep = dm_rep.dot(self.W)

        # Compute the deterministic innovation for the variance
        dv_rep = -norm_cov.dot(sigma_x_rep.T)

        return dm_rep, dv_rep

    def change_pmin_by_innovation(self, x):

        # Compute the change of our posterior at the representer points for
        # different halluzinate function values of x
        dmdb, dvdb = self.innovations(x, self.zb)

        # Update mean and variance of the posterior (at the representer points)
        # by the innovations
        Mb_new = self.Mb + dmdb
        Vb_new = self.Vb + dvdb

        # Return the fantasized pmin
        return mc_part.joint_pmin(Mb_new, Vb_new, self.Nf)




    # def joint_pmin(self, m, V, Nf):
    #     """
    #     Computes the probability of every given point to be the minimum
    #     by sampling function and count how often each point has the
    #     smallest function value.
    #     Parameters
    #     ----------
    #     M: np.ndarray(N, 1)
    #         Mean value of each of the N points.
    #     V: np.ndarray(N, N)
    #         Covariance matrix for all points
    #     Nf: int 
    #         Number of function samples that will be drawn at each point
    #     Returns
    #     -------
    #     np.ndarray(N,1)
    #         pmin distribution
    #     """
    #     Nb = m.shape[0]
    #     noise = 0
    #     while(True):
    #         try:
    #             cV = np.linalg.cholesky(V + noise * np.eye(V.shape[0]))
    #             break
    #         except np.linalg.LinAlgError:

    #             if noise == 0:
    #                 noise = 1e-10
    #             if noise == 10000:
    #                 raise np.linalg.LinAlgError('Cholesky '
    #                     'decomposition failed.')
    #             else:
    #                 noise *= 10

    #     if noise > 0:
    #         logger.error("Add %f noise on the diagonal." % noise)
    #     # Draw new function samples from the innovated GP
    #     # on the representer points
    #     F = np.random.multivariate_normal(mean=np.zeros(Nb), cov=np.eye(Nb), size=Nf)
    #     funcs = np.dot(cV, F.T)
    #     funcs = funcs[:, :, None]

    #     m = m[:, None, :]
    #     funcs = m + funcs

    #     funcs = funcs.reshape(funcs.shape[0], funcs.shape[1] * funcs.shape[2])

    #     # Determine the minima for each function sample
    #     mins = np.argmin(funcs, axis=0)
    #     c = np.bincount(mins)

    #     # Count how often each representer point was the minimum
    #     min_count = np.zeros((Nb,))
    #     min_count[:len(c)] += c
    #     pmin = (min_count / funcs.shape[1])
    #     pmin[np.where(pmin < 1e-70)] = 1e-70

    #     return pmin


    def joint_pmin(self, m, V, Nf):
        """
        Computes the probability of every given point to be the minimum
        by sampling function and count how often each point has the
        smallest function value.
        Parameters
        ----------
        M: np.ndarray(N, 1)
            Mean value of each of the N points.
        V: np.ndarray(N, N)
            Covariance matrix for all points
        Nf: int 
            Number of function samples that will be drawn at each point
        Returns
        -------
        np.ndarray(N,1)
            pmin distribution
        """
        Nb = m.shape[0]
        noise = 0
        while(True):
            try:
                cV = np.linalg.cholesky(V + noise * np.eye(V.shape[0]))
                break
            except np.linalg.LinAlgError:

                if noise == 0:
                    noise = 1e-10
                if noise == 10000:
                    raise np.linalg.LinAlgError('Cholesky '
                        'decomposition failed.')
                else:
                    noise *= 10

        # Draw new function samples from the innovated GP
        # on the representer points
        F = np.random.multivariate_normal(mean=np.zeros(Nb), cov=np.eye(Nb), size=Nf)
        funcs = np.dot(cV, F.T)
        funcs = funcs[:, :, None]

        m = m[:, None, :]
        funcs = m + funcs

        funcs = funcs.reshape(funcs.shape[0], funcs.shape[1] * funcs.shape[2])

        # Determine the minima for each function sample
        mins = np.argmin(funcs, axis=0)
        c = np.bincount(mins)

        # Count how often each representer point was the minimum
        min_count = np.zeros((Nb,))
        min_count[:len(c)] += c
        pmin = (min_count / funcs.shape[1])
        pmin[np.where(pmin < 1e-70)] = 1e-70

        return pmin

        # Nb = m.shape[0] 
        # # cV = np.linalg.cholesky(V)
        # F = np.random.multivariate_normal(mean=np.zeros(Nb), cov=np.eye(Nb), size=Nf)
        # funcs = np.dot(V, F.T)
        # funcs = funcs[:, :, None]

        # m = m[:, None, :]
        # funcs = m + funcs
        # # m1 = m.reshape((Nb,))
        # # try:
        # #     funcs = np.random.multivariate_normal(mean=m1, cov=V, size=Nf)
        # # except np.linalg.LinAlgError:
        # #     print "m1, V, Nf: ", m1, V, Nf
        # # funcs = funcs.T
        # # # print "funcs:", funcs
        # # # funcs = np.dot(cV, F.T)
        # # funcs = funcs[:, :, None]



        # # print "funcs:", funcs.shape

        # funcs = funcs.reshape(funcs.shape[0], funcs.shape[1] * funcs.shape[2])

        # # Determine the minima for each function sample
        # mins = np.argmin(funcs, axis=0)

        # c = np.bincount(mins)

        # # Count how often each representer point was the minimum
        # min_count = np.zeros((Nb,))
        # min_count[:len(c)] += c
        # pmin = (min_count / funcs.shape[1])
        # pmin[np.where(pmin < 1e-70)] = 1e-70

        # # print "funcs:", funcs
        # # print "mins:", mins
        # # print "c:", c
        # # print "min_count: ", min_count
        # # print "pmin:", pmin

        # # print "pmin:", pmin

        # # raw_input("press to continue...")

        # return pmin

    def joint_pmax(self, m, V, Nf):
        """
        Computes the probability of every given point to be the minimum
        by sampling function and count how often each point has the
        smallest function value.
        Parameters
        ----------
        M: np.ndarray(N, 1)
            Mean value of each of the N points.
        V: np.ndarray(N, N)
            Covariance matrix for all points
        Nf: int 
            Number of function samples that will be drawn at each point
        Returns
        -------
        np.ndarray(N,1)
            pmin distribution
        """

        Nb = m.shape[0] 

        # cV = np.linalg.cholesky(V)
        F = np.random.multivariate_normal(mean=np.zeros(Nb), cov=np.eye(Nb), size=Nf)
        
        funcs = np.dot(V, F.T)
        funcs = funcs[:, :, None]

        m = m[:, None, :]
        funcs = m + funcs

        funcs = funcs.reshape(funcs.shape[0], funcs.shape[1] * funcs.shape[2])

        # Determine the minima for each function sample
        maxes = np.argmax(funcs, axis=0)

        c = np.bincount(maxes)

        # Count how often each representer point was the maximum
        max_count = np.zeros((Nb,))
        max_count[:len(c)] += c
        pmax = (max_count / funcs.shape[1])
        pmax[np.where(pmax < 1e-70)] = 1e-70

        return pmax
