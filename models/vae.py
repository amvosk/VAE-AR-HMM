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
class VAEConfig:
    n_channels: int
    n_features: int
    log_transform: bool
    fs: int
    fsamp: int
    ma_order_encoder: int
    ma_order_decoder: int
    use_ma_bias: bool
    ma_bias_common: bool
    use_cov_scaler: bool

    
    


class Rectifier(nn.Module):
    def __init__(self, log_transform=False, logeps=1e-7):
        super(self.__class__,self).__init__()
        self.log_transform = log_transform
        self.eps = logeps
        
    def forward(self, s):
        s_abs = torch.abs(s)
        s_sign = torch.sign(s)
        if self.log_transform:
            s_abs = 2 * torch.log(s_abs + self.eps)
        return s_abs, s_sign
    
    def reverse(self, s_abs, s_sign):
        if self.log_transform:
            s_abs = torch.exp(s_abs / 2) - self.eps
        s = s_abs * s_sign
        return s

    
class Downsampler(nn.Module):
    def __init__(self, in_fs, out_fs):
        super(self.__class__,self).__init__()
        self.downsample_coef = int(in_fs / out_fs)
    
    def forward(self, s_abs):
        s_abs_ = einops.rearrange(s_abs, 'b c (t d) -> b c d t', d=self.downsample_coef)
        a_mean = einops.reduce(s_abs_, 'b c d t -> b c t', 'mean')
        a_mean_ = einops.rearrange(a_mean, 'b c t -> b c 1 t')
        s_centered = s_abs_ - a_mean_
        a_cov = torch.einsum('bcdt,bCdt->bcCt', s_centered, s_centered) / (self.downsample_coef-1)
        return a_mean, a_cov

    
class Upsampler(nn.Module):
    def __init__(self, in_fs, out_fs):
        super(self.__class__,self).__init__()
        self.upsample_coef = int(out_fs / in_fs)
        
    def forward(self, a_mean_tilde, a_cov_tilde=None, cov_propagation=False):
        s_abs = einops.repeat(a_mean_tilde, 'b c t -> b c (t d)', d=self.upsample_coef)
        s_cov = einops.repeat(a_cov_tilde, 'b c C t -> b c C (t d)', d=self.upsample_coef) if cov_propagation else None
        return s_abs, s_cov

    
class MovingAverage(nn.Module):
    def __init__(self, order, n_features, use_bias=True, bias_common=False, random_seed=None):
        super(self.__class__,self).__init__()
        if random_seed is not None:
            torch.manual_seed(random_seed)
        
        self.order = order
        self.n_features = n_features
        
        # self.pad = nn.ConstantPad1d(padding=(order, 0), value=0)
        parameter_shape = (n_features, n_features, order)
        
        
        self.use_bias = use_bias
        self.bias_common = bias_common
        if self.use_bias and not self.bias_common:
            self.bias = nn.Parameter(torch.Tensor(n_features))
            nn.init.constant_(self.bias, 0)
  
        if order > 0:
            self.moving_average = nn.Parameter(torch.Tensor(n_features, n_features, order))
            nn.init.kaiming_uniform_(self.moving_average, a=0.01)
            
            identity = torch.eye(n_features).unsqueeze(-1).to(self.moving_average.device)
            self.register_buffer("identity", identity)

    def forward(self, a_mu, a_Sigma=None, cov_propagation=True, bias=None):
        x_Sigma = None

        if self.order > 0:
            batch_size = a_mu.shape[0]
            n_timestamps = a_mu.shape[-1]

            filtr = torch.cat([self.moving_average, self.identity], dim=-1)

            x_mu = torch.zeros((batch_size, self.n_features, n_timestamps-self.order))
            for i in range(n_timestamps-self.order):
                x_mu[...,i] = torch.einsum('bCt,cCt->bc', a_mu[...,i:i+self.order+1], filtr)

            if cov_propagation:
                x_Sigma = torch.zeros((batch_size, self.n_features, self.n_features, n_timestamps-self.order))
                for i in range(n_timestamps-self.order):
                    x_Sigma[...,i] = torch.einsum('bCSt,cCt,sSt->bcs', a_Sigma[...,i:i+self.order+1], filtr, filtr)
                    
        elif self.order == 0:
            x_mu = a_mu
            if cov_propagation:
                x_Sigma = a_Sigma
                
        if self.use_bias:
            if not self.bias_common:
                bias = self.bias
            bias = einops.rearrange(bias, 'c -> 1 c 1')
            x_mu = x_mu + bias
        return x_mu, x_Sigma
    
    
class SpatialFilter(nn.Module):
    def __init__(self, in_channels, out_channels, random_seed=None):
        super(self.__class__,self).__init__()
        if random_seed is not None:
            torch.manual_seed(random_seed)
            
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))
        nn.init.kaiming_uniform_(self.weight, a=0.01)
        
    def forward(self, mean, cov=None, cov_propagation=False):
        mean = torch.einsum('bit, Oi -> bOt', mean, self.weight)
        cov = torch.einsum('biIt, oi, OI -> boOt', cov, self.weight, self.weight) if cov_propagation else None
        return mean, cov

    
class Encoder(nn.Module):
    def __init__(self, config, random_seed=None):
        super(self.__class__,self).__init__()
        
        self.config = config
        self.ds_coef = int(self.config.fs / self.config.fsamp)
        self.ma_order = self.config.ma_order_encoder + self.config.ma_order_decoder
        
        self.spatial_filter = SpatialFilter(self.config.n_channels, self.config.n_features, random_seed)
        self.rectifier = Rectifier(self.config.log_transform)
        self.downsampler = Downsampler(in_fs=self.config.fs, out_fs=self.config.fsamp)
        self.moving_average = MovingAverage(
            self.config.ma_order_encoder, self.config.n_features, self.config.use_ma_bias, self.config.ma_bias_common, random_seed)

    def forward(self, y_pad, ma_bias=None):
        s_pad, _ = self.spatial_filter(y_pad)
        s_abs_pad, s_sign_pad = self.rectifier(s_pad)
        a_mean_pad, a_cov_pad = self.downsampler(s_abs_pad)
        x_mean_pad, x_cov_pad = self.moving_average(a_mean_pad, a_cov_pad, cov_propagation=True, bias=ma_bias)

        results = {
            's':s_pad[...,self.ma_order*self.ds_coef:],
            's_abs':s_abs_pad[...,self.ma_order*self.ds_coef:],
            's_sign':s_sign_pad[...,self.ma_order*self.ds_coef:],
            'a_mean':a_mean_pad[...,self.ma_order:],
            'a_cov':a_cov_pad[...,self.ma_order:],
            'x_mean_pad':x_mean_pad,
            'x_cov_pad':x_cov_pad,
        }
        return results
    

class Decoder(nn.Module):
    def __init__(self, config, random_seed=None):
        super(self.__class__,self).__init__()
        
        self.config = config
        self.ds_coef = int(self.config.fs / self.config.fsamp)

        self.moving_average = MovingAverage(
            self.config.ma_order_decoder, self.config.n_features, self.config.use_ma_bias, self.config.ma_bias_common, random_seed)
        self.upsampler = Upsampler(in_fs=self.config.fsamp, out_fs=self.config.fs)
        self.rectifier = Rectifier(self.config.log_transform)

        self.spatial_filter = SpatialFilter(self.config.n_features, self.config.n_channels, random_seed)


    def forward(self, x_mean_tilde, x_cov_tilde, s_sign, ma_bias, cov_propagation=False):
        a_mean_tilde, a_cov_tilde = self.moving_average(x_mean_tilde, x_cov_tilde, cov_propagation=cov_propagation, bias=ma_bias)
        results = self.forward_amplitude(a_mean_tilde, a_cov_tilde, s_sign, cov_propagation=cov_propagation)
        results['a_mean_tilde'] = a_mean_tilde
        results['a_cov_tilde'] = a_cov_tilde
        return results
    
    def forward_amplitude(self, a_mean_tilde, a_cov_tilde, s_sign, cov_propagation=False):
        s_abs_tilde, s_cov_tilde = self.upsampler(a_mean_tilde, a_cov_tilde, cov_propagation=cov_propagation)
        s_tilde = self.rectifier.reverse(s_abs_tilde, s_sign)
        y_mean_tilde, y_cov_tilde = self.spatial_filter(s_tilde, s_cov_tilde, cov_propagation=cov_propagation)
        results = {
            's_abs_tilde':s_abs_tilde,
            's_cov_tilde':s_cov_tilde,
            's_tilde':s_tilde,
            'y_mean_tilde':y_mean_tilde,
            'y_cov_tilde':y_cov_tilde,
        }
        return results
    
    
class VAE(nn.Module):
    def __init__(self, config, random_seed=None):
        super(self.__class__,self).__init__()
        
        self.config = config
        self.ds_coef = int(self.config.fs / self.config.fsamp)
        self.ma_order_decoder = self.config.ma_order_decoder
        self.ma_order = self.config.ma_order_encoder + self.config.ma_order_decoder
        
        self.pad_encoder_decoder = nn.ReflectionPad1d(padding=(self.ma_order*self.ds_coef, 0))
        
        if random_seed is not None:
            torch.manual_seed(random_seed)
            
        self.ma_bias = 0
        if self.config.ma_bias_common:
            self.ma_bias = nn.Parameter(torch.Tensor(self.config.n_features))
            nn.init.constant_(self.ma_bias, 0)
        
        self.encoder = Encoder(self.config, random_seed)
        self.decoder = Decoder(self.config, random_seed)

        if self.config.use_cov_scaler:
            self.cov_scaler = nn.Parameter(torch.Tensor(1))
            nn.init.constant_(self.cov_scaler, 1)
    
    def forward(self, y, cov_propagation=False):
        indices = [0]
        seq_length = int(y.shape[-1] / self.ds_coef)
        # y_minibatch, y_mean_tilde, y_cov_tilde, a_mean, a_cov, x_mean, x_cov = self.forward_minibatch(
        #     y, indices, seq_length, cov_propagation=cov_propagation
        # )
        results = self.forward_minibatch(y, indices, seq_length, cov_propagation=cov_propagation)
        
        for k, v in results.items():
            if v is not None:
                v = v.squeeze(0)
            results[k] = v
        return results
    
    def forward_minibatch(self, y, indices, seq_length, cov_propagation=False):
        y_pad = self.pad_encoder_decoder(y)
        y_minibatch_pad = [y_pad[...,index*self.ds_coef:(index+self.ma_order+seq_length)*self.ds_coef] for index in indices]
        y_minibatch_pad = torch.stack(y_minibatch_pad)
        y_minibatch = y_minibatch_pad[...,self.ma_order*self.ds_coef:]

        results_encoder = self.encoder(y_minibatch_pad, ma_bias=self.ma_bias)
        s_sign = results_encoder['s_sign']
        x_mean_pad = results_encoder['x_mean_pad']
        x_cov_pad = results_encoder['x_cov_pad']

        if self.config.use_cov_scaler:
            x_cov_pad = x_cov_pad * self.cov_scaler  
        
        results_decoder = self.decoder(
            x_mean_pad, x_cov_pad, s_sign, ma_bias=-self.ma_bias, cov_propagation=cov_propagation,
        )

        results = {
            'y_minibatch':y_minibatch,
            's':results_encoder['s'],
            's_abs':results_encoder['s_abs'],
            's_sign':s_sign,
            'a_mean':results_encoder['a_mean'],
            'a_cov':results_encoder['a_cov'],
            'x_mean':x_mean_pad[...,self.ma_order_decoder:],
            'x_cov':x_cov_pad[...,self.ma_order_decoder:],
            's_abs_tilde':results_decoder['s_abs_tilde'],
            's_cov_tilde':results_decoder['s_cov_tilde'],
            's_tilde':results_decoder['s_tilde'],
            'y_mean_tilde':results_decoder['y_mean_tilde'],
            'y_cov_tilde':results_decoder['y_cov_tilde'],
        }
        return results

    def reset(self, config=None, random_seed=None):
        if config is not None:
            self.config = config
        if random_seed is not None:
            torch.manual_seed(random_seed)
        
        if self.config.ma_bias_common:
            self.ma_bias = nn.Parameter(torch.Tensor(self.config.n_features))
            nn.init.constant_(self.ma_bias, 0)

        self.encoder = Encoder(self.config, random_seed)
        self.decoder = Decoder(self.config, random_seed)

        if self.config.use_cov_scaler:
            self.cov_scaler = nn.Parameter(torch.Tensor(1))
            nn.init.constant_(self.cov_scaler, 1)
        
    def dkl_loss(self, x_mean, x_cov, x_mean_prior, x_cov_prior):
        x_mean_diff = x_mean - x_mean_prior
        term1 = torch.einsum('bct,bcCt,bCt->bt', x_mean_diff, x_cov_prior, x_mean_diff)

        x_cov_rearranged = einops.rearrange(x_cov, 'b c C t -> b t c C')
        x_cov_prior_rearranged = einops.rearrange(x_cov_prior, 'b c C t -> b t c C')
        cholesky_factor = torch.linalg.cholesky(x_cov_prior_rearranged)
        x_covcov = torch.cholesky_solve(x_cov_rearranged, cholesky_factor)

        x_covcov_diag = torch.diagonal(x_covcov, dim1=2, dim2=3)
        term2 = torch.einsum('btc->bt', x_covcov_diag)

        term3_ = torch.linalg.slogdet(x_covcov)
        term3 = term3_.sign * term3_.logabsdet

        return 1/2 * (term1 + term2 - term3), term1, term2, term3

    def rec_loss(self, y, y_mean_tilde, y_cov_tilde, sigma2y, ds_coef, cov_propagation=False):
        y = einops.rearrange(y, 'b c (t d) -> b t d c', d=ds_coef)
        y_mean_tilde = einops.rearrange(y_mean_tilde, 'b c (t d) -> b t d c', d=ds_coef)
        y_diff = y_mean_tilde - y
        if not cov_propagation:
            y_diff_norm2 = torch.einsum('btdc,btdc->bt', y_diff, y_diff)
            rec_loss = y_diff_norm2 / (2*sigma2y)
        else:
            y_cov_tilde = einops.rearrange(y_cov_tilde, 'b c C (t d) -> b t d c C', d=ds_coef)
            noise_cov = sigma2y * einops.rearrange(torch.eye(y_cov_tilde.shape[-1], device=y_cov_tilde.device), 'c C -> 1 1 1 c C')
            y_cov_tilde = y_cov_tilde + noise_cov
            
            _, logabsdet = torch.linalg.slogdet(y_cov_tilde) # b t d
            logabsdet = torch.einsum('btd -> bt', logabsdet)

            cholesky_factor = torch.linalg.cholesky(y_cov_tilde)
            y_diff_ = einops.rearrange(y_mean_tilde, 'b t d c -> b t d c 1')
            y_diff_scaled_ = torch.cholesky_solve(y_diff_, cholesky_factor)
            y_diff_scaled = einops.rearrange(y_diff_scaled_, 'b t d c 1 -> b t d c')
            
            mahalanobis2 = torch.einsum('btdc, btdc -> bt', y_diff, y_diff_scaled)

            rec_loss = (mahalanobis2 + logabsdet) / 2
        return rec_loss
    
    
    
    
    
    