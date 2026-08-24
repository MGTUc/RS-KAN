#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:11:30 2025

@author: cruz
"""

# model.py
"""
This module defines the model architecture:
1. FullStateNonlinearityKAN: Wraps the efficient_kan.KAN module to model the nonlinear correction.
2. StateSpaceKANModel: Implements the full state-space model by combining linear dynamics
   (with matrices A, B, C, D) and the KAN-based nonlinearity.
"""

import torch
import torch.nn as nn
# from kan import KAN 

class StateSpaceKANModel(nn.Module):
    def __init__(self, A_init, B_init, C_init, D_init, state_kan_model, output_kan_model, dt, trainable_A=True, trainable_B=True, trainable_C=True, trainable_D=True, ):
        """
        Initializes the full state-space model.
        
        Args:
            A_init, B_init, C_init, D_init (Tensor): Initial values for state-space matrices.
            kan_model (nn.Module): The KAN-based nonlinearity module.
            trainable_C (bool): If True, C is learnable.
            trainable_D (bool): If True, D is learnable.
        """
        super(StateSpaceKANModel, self).__init__()
        # self.A = nn.Parameter(A_init.clone().detach(), requires_grad=trainable_A)
        # self.B = nn.Parameter(B_init.clone().detach(), requires_grad=trainable_B)
        # # Set trainability of C and D based on configuration:
        # self.C = nn.Parameter(C_init.clone().detach(), requires_grad=trainable_C)
        # self.D = nn.Parameter(D_init.clone().detach(), requires_grad=trainable_D)


        self.a21 = nn.Parameter(torch.tensor(-3.5497e+01))
        self.a22 = nn.Parameter(torch.tensor(8.4196e-01))
        self.b2  = nn.Parameter(torch.tensor(-2.0555))
        self.register_buffer('C', torch.tensor([[1.0, 0.0]]))
        self.register_buffer('D', torch.tensor([[0.0]]))
        self.register_buffer('A', torch.tensor([[1.0, 0.0], [self.a21.item(), self.a22.item()]]))
        self.register_buffer('B', torch.tensor([[0.0], [self.b2.item()]]))
        self.dt = dt
        
        self.state_dim = self.A.shape[0]
        
        
        self.state_kan_model = state_kan_model
        self.output_kan_model = output_kan_model

    def get_A(self):
        row0 = torch.tensor([1.0, self.dt], device=self.a21.device)
        row1 = torch.stack([self.a21, self.a22])
        return torch.stack([row0, row1])          # rebuilt from current params, every call

    def get_B(self):
        return torch.stack([
            torch.zeros(1, device=self.b2.device),
            self.b2.unsqueeze(0)
        ])                                          # shape (2, 1)

    def forward(self, state, u):
        """
        Computes the next state and output of the system.

        Args:
            state (Tensor): Current state [batch_size, 2]
            u (Tensor): Current input [batch_size, 1]
        
        Returns:
            next_state (Tensor): Next state [batch_size, 2]
            y (Tensor): System output [batch_size, 1]
        """
        # --- State Update ---
        A = self.get_A()
        B = self.get_B()
        linear_next_state = state @ A.T + u @ B.T
        # Apply state KAN if it exists
        if self.state_kan_model:
            state_nonlinear_correction = self.state_kan_model(state=state, u=u)
            next_state = linear_next_state + state_nonlinear_correction
            # pos_std = 3.37e-7   # sqrt(1.1366e-13)
            # vel_std = 1.417e-4  # sqrt(2.0084e-8)

            # state_scaled = state / torch.tensor([pos_std, vel_std])
            # correction_scaled = self.state_kan_model(state=state_scaled, u=u)
            # correction = correction_scaled * torch.tensor([pos_std, vel_std])
            # next_state = linear_next_state + correction
        else:
            # Purely linear state update
            next_state = linear_next_state
            
        # --- Output Calculation ---
        # Calculate linear part of the output equation
        y_linear = state @ self.C.T + u @ self.D.T

        # Apply output KAN if it exists
        if self.output_kan_model:
            output_nonlinear_correction = self.output_kan_model(state=state, u=u)
            y_final = y_linear + output_nonlinear_correction
        else:
            # Purely linear output equation
            y_final = y_linear

        return next_state, y_final

    def plot(self, u, **kwargs):
        """
        Plots the KAN nonlinearities for both state and output if they exist.
        
        Args:
            u (Tensor): Input sequence for simulating the state evolution.
            **kwargs: Additional keyword arguments for the plot function.
        """

        with torch.no_grad():
            current_state = torch.zeros(1, self.state_dim)
            state_list = []
            state_list.append(current_state.clone()) 
            for t in range(len(u)):
                current_input = u[t].unsqueeze(0)
                next_state, _ = self.forward(current_state, current_input)
                state_list.append(next_state.clone())
                current_state = next_state
            state = torch.cat(state_list[1:])
        self.forward(state, u)

        if self.state_kan_model:
            print("Plotting state KAN nonlinearity...")
            self.state_kan_model.plot(**kwargs)
        if self.output_kan_model:
            print("Plotting output KAN nonlinearity...")
            self.output_kan_model.plot(**kwargs)
    
    # --- Add separate regularization loss method ---
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """Computes the combined regularization loss from active KANs."""
        total_reg_loss = 0.0
        if self.state_kan_model and hasattr(self.state_kan_model, 'kan') and hasattr(self.state_kan_model.kan, 'regularization_loss'):
            total_reg_loss += self.state_kan_model.kan.regularization_loss(regularize_activation, regularize_entropy)
        if self.output_kan_model and hasattr(self.output_kan_model, 'kan') and hasattr(self.output_kan_model.kan, 'regularization_loss'):
             total_reg_loss += self.output_kan_model.kan.regularization_loss(regularize_activation, regularize_entropy)
        return total_reg_loss

    def get_corrections_only(self, state, u,):
        """
        proxy of forward to obtain non-linear contributions separatly
        """
        # --- State Update ---
        
        linear_next_state = state @ self.A.T + u @ self.B.T
        # Apply state KAN if it exists
        if self.state_kan_model:
            state_nonlinear_correction = self.state_kan_model(state=state, u=u)
            next_state = linear_next_state + state_nonlinear_correction
        else:
            # Purely linear state update
            state_nonlinear_correction = 0
            next_state = linear_next_state
            
        # --- Output Calculation ---
        # Calculate linear part of the output equation
        y_linear = state @ self.C.T + u @ self.D.T

        # Apply output KAN if it exists
        if self.output_kan_model:
            output_nonlinear_correction = self.output_kan_model(state=state, u=u)
            y_final = y_linear + output_nonlinear_correction
        else:
            # Purely linear output equation
            output_nonlinear_correction = 0
            y_final = y_linear
        return state_nonlinear_correction, output_nonlinear_correction