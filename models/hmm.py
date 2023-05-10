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

class InitState:
    def __init__(self, n_states, init_state_type='uniform', random_seed=None):
        self.n_states = n_states
        self.init_state_type = init_state_type
        self.random_seed = random_seed

    def get_init_state(self):
        if self.init_state_type == 'uniform':
            init_state = torch.ones(self.n_states) / self.n_states
        elif self.init_state_type == 'random':
            if self.random_seed is not None:
                torch.manual_seed(self.random_seed)
            init_state = torch.rand(self.n_states) 
            init_state = init_state / torch.sum(init_state, dim=0)
        elif self.init_state_type == 'first':
            init_state = torch.zeros(self.n_states)
            eps=1e-7
            init_state[0] = 1 - eps*(self.n_states-1)
            init_state[1:] = eps
        return init_state.detach()
        
def test_InitState(n_states=5, init_state_type='uniform'):
    sp = InitState(n_states=n_states, init_state_type=init_state_type)
    print(sp.get_init_state())
    
        
class TransitionMatrix:
    def __init__(self, n_states, transition_matrix_type, transition_type='multiple', 
                 diagonal_coef=0, first_coef=0, ergodicity_coef=0, random_seed=None):
        self.n_states = n_states
        self.transition_matrix_type = transition_matrix_type
        self.transition_type = transition_type
        self.diagonal_coef = diagonal_coef
        self.first_coef = first_coef
        self.ergodicity_coef = ergodicity_coef
        self.random_seed = random_seed

    def get_transition_matrix(self):
        transition_mask = self.get_transition_mask()
        
        if self.transition_matrix_type == 'uniform':
            transition_matrix = torch.ones((self.n_states, self.n_states))
        elif self.transition_matrix_type == 'random':
            if self.random_seed is not None:
                torch.manual_seed(self.random_seed)
            transition_matrix = torch.rand((self.n_states, self.n_states))
        transition_matrix = transition_matrix * transition_mask

        if self.diagonal_coef > 0:
            transition_matrix += torch.eye(self.n_states) * self.diagonal_coef
        if self.first_coef > 0:
            transition_matrix[0,0] += 1*self.first_coef
        if self.ergodicity_coef > 0:
            transition_matrix_multiplier = torch.sum(transition_matrix, dim=1, keepdims=True)
            transition_matrix += torch.ones((self.n_states, self.n_states)) * self.ergodicity_coef * transition_matrix_multiplier

        transition_matrix = transition_matrix / torch.sum(transition_matrix, dim=1, keepdims=True)
        return transition_matrix.detach()
        
    def get_transition_mask(self):
        if self.transition_type == 'full':
            transition_mask = torch.ones((self.n_states, self.n_states))
        elif self.transition_type == 'single':
            transition_mask = torch.eye(self.n_states)
            transition_mask[:-1,1:] += torch.eye(self.n_states-1)
            transition_mask[:,0] = 1
        elif self.transition_type == 'multiple':
            transition_mask = torch.eye(self.n_states)
            for i in range(1, self.n_states):
                transition_mask[:-i,i:] += torch.eye(self.n_states-i)
            transition_mask[:,0] = 1
        return transition_mask
    
def test_TransitionMatrix(n_states=5, transition_matrix_type='uniform', transition_type='multiple', 
                          diagonal_coef=1, ergodicity_coef=0.001, random_seed=0):
    tm = TransitionMatrix(n_states=n_states, transition_matrix_type=transition_matrix_type, transition_type=transition_type, 
                          diagonal_coef=diagonal_coef, ergodicity_coef=ergodicity_coef, random_seed=random_seed)
    print(tm.get_transition_matrix())



@dataclass
class HMMConfig:
    n_states: int
    init_state_type: str
    transition_matrix_type: str
    transition_type: str
    diagonal_coef: float
    first_coef: float
    ergodicity_coef: float
    temperature: float
    
    
class HMM(nn.Module):
    def __init__(self, config, random_seed=0):
        super(self.__class__,self).__init__()
        self.config = config

        init_state_helper = InitState(
            self.config.n_states, 
            self.config.init_state_type, 
            random_seed,
        )
        init_state = init_state_helper.get_init_state()
        self.register_buffer("init_state", init_state)

        transition_helper = TransitionMatrix(
            self.config.n_states, 
            self.config.transition_matrix_type, 
            self.config.transition_type, 
            self.config.diagonal_coef, 
            self.config.first_coef, 
            self.config.ergodicity_coef, 
            random_seed,
        )
        transition_matrix = transition_helper.get_transition_matrix()
        self.register_buffer("transition_matrix", transition_matrix)
        
        
    def forward(self, x, log_gamma):
        # features = self.feature_extractor(x)
        # loss = self.autoregression(features, torch.exp(log_gamma))
        return None

    def alpha_calculation(self, emissions): 
        batch_size, T_max = emissions.shape[0], emissions.shape[-1]
        log_alpha = torch.zeros((batch_size, self.config.n_states, T_max), device=emissions.device)
        
        log_transition_matrix = einops.rearrange(torch.log(self.transition_matrix), 'i j -> 1 i j')
        for t in range(T_max):
            if t == 0:
                transition = einops.rearrange(torch.log(self.init_state), 'i -> 1 i')
            else:
                # transition = log_transition_matrix + einops.rearrange(log_alpha[...,t-1], 'b i -> b i 1')
                transition = log_transition_matrix + log_alpha[...,t-1].unsqueeze(-1)
                transition = torch.logsumexp(transition, dim=-2)
            log_alpha[...,t] = emissions[...,t] + transition
        return log_alpha
    
    def beta_calculation(self, emissions):
        batch_size, T_max = emissions.shape[0], emissions.shape[-1]
        log_beta = torch.zeros((batch_size, self.config.n_states, T_max), device=emissions.device)
        
        log_transition_matrix = einops.rearrange(torch.log(self.transition_matrix), 'i j -> 1 i j')
        for t in list(range(T_max-1))[::-1]:
            # emission = einops.rearrange(emissions[...,t+1], 'b j -> b 1 j')
            # log_beta_ = einops.rearrange(log_beta[...,t+1], 'b j -> b 1 j')
            emission = emissions[...,t+1].unsqueeze(-2)
            log_beta_ = log_beta[...,t+1].unsqueeze(-2)
            log_beta_zj = log_transition_matrix + emission + log_beta_
            log_beta[...,t] = torch.logsumexp(log_beta_zj, dim=-1)
        return log_beta
    
    def gamma_calculation(self, log_alpha, log_beta, mask):
        log_gamma_unnormalized = log_alpha + log_beta
        log_gamma_normalization = torch.logsumexp(log_gamma_unnormalized, dim=-2, keepdim=True)
        log_gamma = log_gamma_unnormalized - log_gamma_normalization
        return log_gamma, log_gamma_normalization
    
    def xi_calculation(self, log_alpha, log_beta, emissions, mask=None):
        batch_size, T_max = emissions.shape[0], emissions.shape[-1]
        if mask is None: mask = -1 * torch.ones(T_max, device=emissions.device)

        left = einops.rearrange(log_alpha[...,:-1], 'b i L -> b i 1 L')
        right = einops.rearrange(emissions[...,1:], 'b j L -> b 1 j L')
        right += einops.rearrange(log_beta[...,1:], 'b j L -> b 1 j L')
        middle = einops.repeat(torch.log(self.transition_matrix), 'i j -> 1 i j L', L=T_max-1)

        log_xi = left + middle + right

        marginal_likelihood = torch.logsumexp(log_xi, dim=(-3, -2), keepdim=True)
        log_xi = log_xi - marginal_likelihood
        return log_xi
    
    
    def fit(self, emissions, mask=None):
        if len(emissions.shape) < 3:
            emissions = einops.rearrange(emissions, 'f t -> 1 f t')

        T_max = emissions.shape[-1]
        log_alpha = self.alpha_calculation(emissions)
        log_beta = self.beta_calculation(emissions)
        log_gamma, _ = self.gamma_calculation(log_alpha, log_beta, mask)
        log_xi = self.xi_calculation(log_alpha, log_beta, emissions, mask)
        
        log_init_state = log_gamma[...,0]
        init_state = torch.exp(log_init_state)**self.config.temperature
        init_state = torch.mean(init_state, dim=0)
        self.init_state = init_state / torch.sum(init_state)
        
        log_transition_matrix = torch.logsumexp(log_xi, dim=-1) - torch.logsumexp(log_gamma[...,:-1], dim=-1).unsqueeze(-1)
        transition_matrix = torch.exp(log_transition_matrix)**self.config.temperature
        transition_matrix = torch.mean(transition_matrix, dim=0)
        self.transition_matrix = transition_matrix / torch.sum(transition_matrix, dim=1, keepdims=True)

        log_prob = torch.logsumexp(log_alpha[...,-1], dim=-1)
        log_prob = torch.mean(log_prob, dim=0)
        gamma = torch.exp(log_gamma)
        gamma = torch.mean(gamma, dim=0)
        return log_prob, gamma
    
        
    def predict(self, emissions, mask=None):
        T_max = emissions.shape[-1]
        log_delta = torch.zeros((self.config.n_states, T_max), device=emissions.device)
        psi = torch.zeros((self.config.n_states, T_max), device=emissions.device)
        path = torch.zeros(T_max, dtype=torch.long, device=emissions.device)

        log_transition_matrix = torch.log(self.transition_matrix)
        for t in range(T_max):
            emission = emissions[...,t]
            
            if t == 0:
                transition, backtrack = torch.log(self.init_state), 0
            else:
                transition_ = log_transition_matrix + einops.rearrange(log_delta[...,t-1], 'i -> i 1')
                transition, backtrack = torch.max(transition_, dim=0)[0], torch.argmax(transition_, dim=0)
            log_delta[...,t] = emission + transition
            psi[...,t] = backtrack

        path[-1] = torch.argmax(log_delta[...,-1], dim=0)
        for t in list(range(T_max-1))[::-1]:
            path[t] = psi[path[t+1], t+1]
        return path
    
    def reset(self, config=None, random_seed=0):
        if config is not None:
            self.config = config
        
        init_state_helper = InitState(
            self.config.n_states, 
            self.config.init_state_type, 
            random_seed
        )
        self.init_state = init_state_helper.get_init_state()

        transition_helper = TransitionMatrix(
            self.config.n_states, 
            self.config.transition_matrix_type, 
            self.config.transition_type, 
            self.config.diagonal_coef, 
            self.config.first_coef, 
            self.config.ergodicity_coef, 
            random_seed,
        )
        self.transition_matrix = transition_helper.get_transition_matrix()