import os
import time
import math

import numpy as np

import matplotlib.pyplot as plt
import scipy.signal as sg
import sklearn.preprocessing
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from dataclasses import dataclass

    
    
@dataclass
class ARConfig:
    n_states: int
    n_features: int
    ar_order: int
    covariance_type: str


class Autoregression(nn.Module):
    def __init__(self, config):
        super(self.__class__,self).__init__()
        self.config = config
        # self.n_states = config.n_states
        # self.n_features = config.n_features
        # self.ar_order = config.ar_order
        
        self.pad = nn.ConstantPad1d(padding=(self.config.ar_order, 0), value=0)
        
        An = torch.zeros((self.config.n_states, self.config.ar_order + 2, self.config.n_features, self.config.n_features))
        An[:,0,:,:] = - einops.repeat(torch.eye(self.config.n_features), 'c C -> k c C', k=self.config.n_states)
        self.register_buffer("An", An)
        
        Sigma = einops.repeat(torch.eye(self.config.n_features), 'c C -> k c C', k=self.config.n_states)
        self.register_buffer("Sigma", Sigma)
        
        self.whitening_filters = torch.nn.ModuleList()
        for k in range(self.config.n_states):
            conv1d = torch.nn.Conv1d(self.config.n_features, self.config.n_features, kernel_size=self.config.ar_order+1)
            for param in conv1d.parameters():
                param.requires_grad = False
            self.whitening_filters.append(conv1d)
        self._update_filters()
        
    def _update_filters(self):
        An = self._get_ar_coef()
        mean = self._get_mean()
        for k in range(self.config.n_states):
            Akn = torch.flip(An[k], dims=(0,))
            Akn = einops.rearrange(Akn, 'n c C -> c C n')
            if self.config.ar_order > 0:
                self.whitening_filters[k].weight.data[:,:,:-1] = - Akn
            self.whitening_filters[k].weight.data[:,:,-1] = torch.eye(self.config.n_features, device=Akn.device)
            self.whitening_filters[k].bias.data = - mean[k]

        
    def _cholesky_inverse(self, matrix):
        cholesky_factor = torch.linalg.cholesky(matrix)
        matrix_inverse = torch.cholesky_inverse(cholesky_factor)
        return matrix_inverse

    def fit(self, x_mean, x_cov, gamma=None):
        if len(x_mean.shape) == 3:
            x_mean = einops.rearrange(x_mean, '1 c t -> c t')
        if len(x_cov.shape) == 4:
            x_cov = einops.rearrange(x_cov, '1 c C t -> c C t')
        
        n_samples = x_mean.shape[-1]
        pad = torch.nn.ConstantPad1d(padding=(self.config.ar_order, 0), value=0)
        x_mean_pad = self.pad(x_mean)
        x_cov_pad = self.pad(x_cov)
    
        if gamma is None:
            gamma = torch.ones((self.config.n_states, n_samples), device=x_mean.device) / self.config.n_states
        self.An[:,1:,:,:] = torch.clone(self._estimate_autoregression(x_mean, x_cov, gamma))
        Sigma = torch.clone(self._estimate_covariance(x_mean, x_cov, gamma))
        self.Sigma = self._transform_covariance(Sigma)
        self._update_filters()
        
    def _signal_whitening(self, x_mean):
        es = torch.zeros(self.config.n_states, *x_mean.size(), device=x_mean.device)
        for k in range(self.config.n_states):
            es[k] = self.whitening_filters[k](self.pad(x_mean))
        return es
    
    def emission_log_prob(self, x_mean):
        if len(x_mean.shape) < 3:
            x_mean = einops.rearrange(x_mean, 'c t -> 1 c t')
        n_samples = x_mean.shape[-1]
        es = self._signal_whitening(x_mean)
        # print('es', es.shape)
        
        Dlog2pi = self.config.n_features * math.log(2*math.pi)
        
        _, logabsdet = torch.linalg.slogdet(self.Sigma)
        logabsdet = einops.rearrange(logabsdet, 'k -> 1 k 1')

        cholesky_factor = torch.linalg.cholesky(self.Sigma)
        cholesky_factor = einops.rearrange(cholesky_factor, 'k c C -> k 1 c C')
        # print('cholesky_factor', cholesky_factor.shape)
        es_scaled = torch.cholesky_solve(es, cholesky_factor)
        # print('es_scaled', es_scaled.shape)
        mahalanobis = torch.einsum('kbct, kbct -> bkt', es, es_scaled)
        
        log_prob = - 1/2 * (Dlog2pi + logabsdet + mahalanobis)
        return log_prob
        
    def emission_log_prob_pytorch(self, x_mean):
        if len(x_mean.shape) < 3:
            x_mean = einops.rearrange(x_mean, 'c t -> 1 c t')
        from torch.distributions import MultivariateNormal
        
        log_prob = torch.zeros((1, self.config.n_states, x_mean.shape[-1]), device=x_mean.device)
        es = self._signal_whitening(x_mean)
        es = einops.rearrange(es, 'k b c t -> b k t c')

        for k in range(self.config.n_states):
            # covariance_matrix_k = self.Sigma[k]
            # covariance_matrix_k = einops.rearrange(covariance_matrix_k, 'c C -> 1 k c C')
            distribution = MultivariateNormal(torch.zeros((1, self.config.n_features), device=x_mean.device), self.Sigma[k])
            log_prob[k] = distribution.log_prob(es[k])
        return log_prob
    
    def get_posterior_base_slow(self, x_mean, gamma):
        n_samples = x_mean.shape[-1]
        x_mean_pad = self.pad(x_mean)
        
        cov_hat_inv = self._cholesky_inverse(self.Sigma)
        gcov_hat_inv = torch.einsum('kt,kcC->tcC', gamma, cov_hat_inv)
        gcov_hat = self._cholesky_inverse(gcov_hat_inv)

        An_filt = torch.flip(self.An[:,1:], dims=(1,))

        gmean_hat = torch.zeros((n_samples, self.config.n_features), device=gcov_hat.device)
        for t in range(n_samples):
            x_mean_chunk = x_mean_pad[...,t:t+self.config.ar_order]
            x_mean_chunk = torch.cat([torch.ones((self.config.n_features,1), device=x_mean_chunk.device), x_mean_chunk], dim=-1)
            gmean_hat[t] = torch.einsum('cC, k, kCs, knsS, Sn -> c', gcov_hat[t], gamma[...,t], cov_hat_inv, An_filt, x_mean_chunk)
        return gmean_hat, gcov_hat
    
    def get_posterior_base(self, x_mean, gamma):
        n_samples = x_mean.shape[-1]
        x_mean_pad = self.pad(x_mean)
        
        cov_hat_inv = self._cholesky_inverse(self.Sigma)
        gcov_hat_inv = torch.einsum('kt,kcC->tcC', gamma, cov_hat_inv)
        gcov_hat = self._cholesky_inverse(gcov_hat_inv)
        gcov_hat = einops.rearrange(gcov_hat, 't c C -> c C t')
        
        Amun = torch.zeros(
            (self.config.n_states, self.config.ar_order + 1, self.config.n_features, n_samples), 
            device=x_mean_pad.device
        )
        for i in range(self.config.ar_order):
            Amun[:,i,...] = torch.einsum(
                'kcC, Ct -> kct', self.An[:,i], x_mean_pad[:,self.config.ar_order-i:n_samples+self.config.ar_order-i]
            )
        Amun[:,-1,...] = einops.repeat(torch.diagonal(self.An[:,-1], dim1=-2, dim2=-1), 'k c -> k c t', t=n_samples)
        Amun = torch.einsum('knCt -> kCt', Amun)
        
        gmean_hat = torch.einsum('sct, kt, kcC, kCt -> st', gcov_hat, gamma, cov_hat_inv, Amun)
        return gmean_hat, gcov_hat
    
    def _get_ar_coef(self):
        return torch.clone(self.An[:,1:-1])
    def _get_mean(self):
        return torch.clone(torch.diagonal(self.An[:,-1], dim1=-2, dim2=-1))
    def _get_cov(self):
        return torch.clone(self.Sigma)
            
    def _estimate_autoregression(self, x_mean, x_cov, gamma):
        An_hat = torch.zeros((self.config.n_states, self.config.ar_order + 1, self.config.n_features, self.config.n_features), device=x_mean.device)
        x_meanT = einops.rearrange(x_mean, 'c t -> t c')
        x_covT = einops.rearrange(x_cov, 'c C t -> t c C')
        

        N, D = self.config.ar_order, self.config.n_features
        if N > 0:
            covn = [torch.cat((torch.zeros((n, self.config.n_features, self.config.n_features), device=x_mean.device), x_covT[:-n]), dim=0) for n in range(1, N+1)]
            covn = torch.stack(covn)
            # print(gamma.shape, covn.shape)
            gcov = torch.einsum('kt,ntcC->kncC', gamma, covn)
        
        for k in range(self.config.n_states):
            g = einops.rearrange(gamma[k], 't -> t 1')
            xn = [x_meanT] + [torch.cat((torch.zeros((n, self.config.n_features), device=x_mean.device), x_meanT[:-n]), dim=0) for n in range(1, N+1)]
            xng = [g*xn_ for xn_ in xn]

            B = torch.sum(xng[0], dim=0, keepdims=True)
            if N > 0:
                B = torch.cat([(xn[n].T @ xng[0]) for n in range(1, N+1)] + [B], dim=0)

            M = torch.zeros((N*D+1, N*D+1), device=x_mean.device)
            for i in range(1, N+1):
                x_ = torch.sum(xng[i], dim=0)
                M[-1,(i-1)*D:i*D] = x_
                M[(i-1)*D:i*D,-1] = x_
            M[-1,-1] = torch.sum(g)

            for i in range(1, N+1):
                for j in range(1, N+1):
                    if j > i: 
                        xgx = xn[i].T @ xng[j]
                        M[D*(i-1):D*i, D*(j-1):D*j] = xgx
                        M[D*(j-1):D*j, D*(i-1):D*i] = xgx.T
                    elif j == i:
                        xgx = xn[i].T @ xng[j]
                        M[D*(i-1):D*i, D*(j-1):D*j] = xgx + gcov[k,i-1]

            cholesky_factor = torch.linalg.cholesky(M)
            A = torch.cholesky_solve(B, cholesky_factor)

            if N > 0:
                An_hat[k,:-1] = einops.rearrange(A[:-1], '(N D) d -> N d D', N=N)
            An_hat[k,-1] = torch.diag(A[-1])
        return An_hat
    
    def _estimate_covariance_slow(self, x_mean, x_cov, gamma):
        n_samples = x_mean.shape[-1]
        x_mean_pad = self.pad(x_mean)
        x_cov_pad = self.pad(x_cov)

        An_mean = torch.flip(self.An, dims=(1,))
        An_cov = torch.flip(self.An[:,:-1], dims=(1,))

        cov = torch.zeros((self.config.n_states, self.config.n_features, self.config.n_features, n_samples), device=x_mean_pad.device)
        for t in range(n_samples):
            x_mean_chunk = x_mean_pad[...,t:t+self.config.ar_order+1]
            x_mean_chunk = torch.cat([torch.ones((self.config.n_features,1), device=x_mean_chunk.device), x_mean_chunk], dim=-1)
            x_cov_chunk = x_cov_pad[...,t:t+self.config.ar_order+1]
            left = torch.einsum('kncC, CSn, knsS -> kcs', An_cov, x_cov_chunk, An_cov)
            right = torch.einsum('kncC, Cn, Sm, kmsS -> kcs', An_mean, x_mean_chunk, x_mean_chunk, An_mean)
            cov[...,t] = left + right

        gamma_sum = einops.rearrange(torch.einsum('tk->k', gamma), 'k -> k 1 1')
        cov_hat = torch.einsum('tk,kcCt->kcC', gamma, cov) / gamma_sum
        return cov_hat
    
    def _estimate_covariance(self, x_mean, x_cov, gamma):
        n_samples = x_mean.shape[-1]
        x_mean_pad = self.pad(x_mean)
        x_cov_pad = self.pad(x_cov)

        Amun = torch.zeros(
            (self.config.n_states, self.config.ar_order + 2, self.config.n_features, n_samples), 
            device=x_mean_pad.device
        )
        ASigman = torch.zeros(
            (self.config.n_states, self.config.ar_order + 1, self.config.n_features, self.config.n_features, n_samples), 
            device=x_cov_pad.device
        )
        
        for i in range(self.config.ar_order + 1):
            ASigman[:,i,...] = torch.einsum(
                'kcs, kCS, sSt -> kcCt', self.An[:,i], self.An[:,i], x_cov_pad[...,self.config.ar_order-i:n_samples+self.config.ar_order-i]
            )
            Amun[:,i,...] = torch.einsum(
                'kcC, Ct -> kct', self.An[:,i], x_mean_pad[:,self.config.ar_order-i:n_samples+self.config.ar_order-i]
            )
        Amun[:,-1,...] = einops.repeat(torch.diagonal(self.An[:,-1], dim1=-2, dim2=-1), 'k c -> k c t', t=n_samples)
        right = torch.einsum('kict, kjCt -> kijcCt', Amun, Amun)
        right = torch.einsum('kijcCt -> kcCt', right)
        left = torch.einsum('kicCt -> kcCt', ASigman)
        Sigma = left + right
        
        gamma_sum = einops.rearrange(torch.einsum('kt->k', gamma), 'k -> k 1 1')
        Sigma_hat = torch.einsum('kt,kcCt->kcC', gamma, Sigma) / gamma_sum
        return Sigma_hat
    
    def _transform_covariance(self, Sigma_hat):
        if self.config.covariance_type == 'full':
            return Sigma_hat
        elif self.config.covariance_type == 'diagonal':
            identity = einops.repeat(torch.eye(self.config.n_features, device=Sigma_hat.device), 'c C -> k c C', k=self.config.n_states)
            return Sigma_hat * identity
        elif self.config.covariance_type == 'spherical':
            identity = einops.repeat(torch.eye(self.config.n_features, device=Sigma_hat.device), 'c C -> k c C', k=self.config.n_states)
            diagonal = torch.diagonal(Sigma_hat, dim1=-2, dim2=-1)
            diagonal_mean = einops.rearrange(torch.mean(diagonal, dim=-1), 'k -> k 1 1')
            return diagonal_mean * identity
        elif self.config.covariance_type == 'common':
            Sigma_mean = einops.repeat(torch.mean(Sigma_hat, dim=0), 'c C -> k c C', k=self.config.n_states)
            return Sigma_mean
        elif self.config.covariance_type == 'identity':
            identity = einops.repeat(torch.eye(self.config.n_features, device=Sigma_hat.device), 'c C -> k c C', k=self.config.n_states)
            return identity
        
    def sample(self, x_mean, x_cov, n_samples, random_seed=None):
        if random_seed is not None:
            torch.manual_seed(random_seed)
        if len(x_mean.shape) < 3:
            x_mean = einops.rearrange(x_mean, 'c t -> 1 c t')
        if len(x_cov.shape) < 4:
            x_cov = einops.rearrange(x_cov, 'c C t -> 1 c C t')
        eps = torch.randn(size=(n_samples, *x_mean.shape), device=x_mean.device)
        x_cov_reshaped = einops.rearrange(x_cov, 'b c C t -> b t c C')
        cholesky_factor = torch.linalg.cholesky(x_cov_reshaped)
        x_samples = torch.einsum('btcC, sbCt -> sbct', cholesky_factor, eps)
        x_samples = x_samples + einops.repeat(x_mean, 'b c t -> s b c t', s=n_samples)
        x_samples = einops.rearrange(x_samples, 's b c t -> (s b) c t')
        return x_samples
    
    
    def reset(self):
        self.An = torch.zeros(
            (self.config.n_states, self.config.ar_order + 2, self.config.n_features, self.config.n_features),
            device=self.An.device
        )
        self.An[:,0,:,:] = - einops.repeat(torch.eye(self.config.n_features), 'c C -> k c C', k=self.config.n_states)
        self.Sigma = einops.repeat(torch.eye(self.config.n_features, device=self.Sigma.device), 'c C -> k c C', k=self.config.n_states)
        self._update_filters()