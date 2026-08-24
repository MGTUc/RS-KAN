#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 10:49:20 2025

@author: cruz
"""
import matplotlib.pyplot as plt
import torch
import numpy as np
import silverbox._utils as _utils
import nonlinear_benchmarks 
from scipy import signal
import scipy.io

class SystemIdentificationDataset:
    def __init__(self, test_case='Silverbox', test_flag=None, norm_flag='minmax',device='cpu',
                 states_available=False,init_matrices_flag=None):
        """
        Loads and preprocesses system identification datasets.

        Args:
            test_case (str): 'Silverbox' or 'Wiener-Hammerstein'.
            test_flag (str, optional): Specific test flag for 'Silverbox' (e.g., 'arrow_extra').
            norm_flag (str): Normalization type ('minmax', 'zscore', etc.).
        """
        self.states_available=states_available
        self.init_matrices_flag = init_matrices_flag
        self.device=device
        self.test_case = test_case
        self.test_flag = test_flag
        self.norm_flag = norm_flag
        self.dt = None # Sampling time
        self.x_min, self.x_max = None, None # Normalization parameters for output/state
        self.x_dot_min, self.x_dot_max = None, None # Normalization parameters for state derivative
        self.u_min, self.u_max = None, None # Normalization parameters for input
        self.X_train_norm, self.u_train_norm, self.y_train_norm = None, None, None # Normalized training data
        self.X_test_norm, self.u_test_norm, self.y_test_norm = None, None, None # Normalized test data

        self.X_dim  = 0
        self.u_dim  = 0
        self.y_dim  = 0

        self.A_init, self.B_init, self.C_init, self.D_init = None, None, None, None # For StateSpaceModel

        self._load_and_preprocess() # Call the data loading and preprocessing method in the constructor
        self._init_lin_matrices() 
        
    def _load_and_preprocess(self):
        test_case = self.test_case
        test_flag = self.test_flag
        norm_flag = self.norm_flag

        if self.test_case == 'Silverbox':
            train_val, test = nonlinear_benchmarks.Silverbox(atleast_2d=True)
            self.dt = train_val.sampling_time
            f_s = 1 / self.dt
            u_train, y_train = train_val

        elif self.test_case == 'Wiener-Hammerstein':
            train_val, test = nonlinear_benchmarks.WienerHammerBenchMark(atleast_2d=True)
            self.dt = train_val.sampling_time
            print(test.state_initialization_window_length)
            u_train, y_train = train_val
            u_test, y_test = test


        elif self.test_case == 'Luca-Airfoil-CFD':
            mat_CFD = scipy.io.loadmat('TimeSeries_Exp_CFD/DatasetCFD.mat')

            u_train = mat_CFD['uTrain'] # Angle of attack
            y_train = mat_CFD['yTrain'] # Lift coefficient
            f_s = 200
            self.dt = 1/f_s
            u_test = mat_CFD['uTrain'] # Angle of attack
            y_test = mat_CFD['yTrain'] # Lift coefficient
            
        elif self.test_case == 'Luca-Airfoil-CFD-pitchRate':
            mat_CFD = scipy.io.loadmat('TimeSeries_Exp_CFD/DatasetCFD.mat')

            u_train = mat_CFD['uTrain'] # Angle of attack
            y_train = mat_CFD['yTrain'] # Lift coefficient
            f_s = 200
            self.dt = 1/f_s
            u_test = mat_CFD['uTrain'] # Angle of attack
            y_test = mat_CFD['yTrain'] # Lift coefficient
            self.time = np.arange(0,32008*self.dt,self.dt)
            #very raw approach
            da_dt_train = np.gradient(u_train.ravel(),self.time)
            da_dt_train[da_dt_train > 200] = 0
            u_train = np.concatenate((u_train,da_dt_train.reshape(-1,1)),axis=1)
            da_dt_test = da_dt_train # change in future!
            u_test = np.concatenate((u_test,da_dt_test.reshape(-1,1)),axis=1)
            # I think I might have to remake Luca function and create the data since the function he uses is symbolic differentable?
            
            
        elif self.test_case == 'Luca-Airfoil-Exp':
            mat_exp = scipy.io.loadmat('/Users/cruz/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/Python_scripts/ss-kan-paper/TimeSeries_Exp_CFD/DatasetExp_LowTI.mat')
            u_train = mat_exp['aoa'] # Angle of attack
            y_train = mat_exp['cl'] # Lift coefficient
            f_s = 200
            self.dt = 1/f_s
            u_test = mat_exp['aoa'] # Angle of attack
            y_test = mat_exp['cl'] # Lift coefficient
            self.time = np.arange(0,160000*self.dt,self.dt)
            
        elif self.test_case == 'Luca-Airfoil-Exp-pitchRate':
            mat_exp = scipy.io.loadmat('/Users/cruz/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/Python_scripts/ss-kan-paper/TimeSeries_Exp_CFD/DatasetExp_LowTI.mat')
            u_train = mat_exp['aoa'] # Angle of attack
            y_train = mat_exp['cl'] # Lift coefficient
            f_s = 200
            self.dt = 1/f_s
            u_test = mat_exp['aoa'] # Angle of attack
            y_test = mat_exp['cl'] # Lift coefficient
            self.time = np.arange(0,160000*self.dt,self.dt)
            #very raw approach
            da_dt_train = np.gradient(u_train.ravel(),self.time)
            da_dt_train[da_dt_train > 200] = 0
            u_train = np.concatenate((u_train,da_dt_train.reshape(-1,1)),axis=1)
            da_dt_test = da_dt_train # change in future!
            u_test = np.concatenate((u_test,da_dt_test.reshape(-1,1)),axis=1)
            # I think I might have to remake Luca function and create the data since the function he uses is symbolic differentable?
            
            
        else:
            raise ValueError(f"Unknown test_case: {test_case}")

        u_train = torch.tensor(u_train, dtype=torch.float32,device=self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32,device=self.device)
        t_train = torch.arange(len(y_train), dtype=torch.float32,device=self.device) * self.dt
        self.t_train = t_train
        y_train_norm, self.x_min, self.x_max, _ = _utils.normalize_data(y_train, norm_type=norm_flag, normalize=True)
        u_train_norm, self.u_min, self.u_max, _ = _utils.normalize_data(u_train, norm_type=norm_flag, normalize=True)


        if self.states_available is True:
            x_dot_train = torch.zeros_like(y_train,device=self.device)
            x_dot_train[1:-1] = (y_train[2:] - y_train[:-2]) / (2 * self.dt)
            x_dot_train[0] = (y_train[1] - y_train[0]) / self.dt
            x_dot_train[-1] = (y_train[-1] - y_train[-2]) / self.dt
            x_dot_train_norm, self.x_dot_min, self.x_dot_max, _ = _utils.normalize_data(x_dot_train, norm_type=norm_flag, normalize=True)
            X_train_norm = torch.cat((y_train_norm, x_dot_train_norm), dim=-1)
            self.X_train_norm = X_train_norm
            
        # Process test data
        if test_case=='Silverbox':
            if test_flag=='arrow_no_extra':
                test_multisine, test_arrow_full, test_arrow_no_extrapolation = test
                u_test, y_test = test_arrow_no_extrapolation.u, test_arrow_no_extrapolation.y
            elif test_flag=='arrow_extra':
                test_multisine, test_arrow_full, test_arrow_no_extrapolation = test
                u_test, y_test = test_arrow_full.u, test_arrow_full.y
            elif test_flag=='multisine':
                test_multisine, test_arrow_full, test_arrow_no_extrapolation = test
                u_test, y_test = test_multisine.u, test_multisine.y


        u_test = torch.tensor(u_test, dtype=torch.float32,device=self.device)
        y_test = torch.tensor(y_test, dtype=torch.float32,device=self.device)
        t_test = torch.arange(len(y_test), dtype=torch.float32,device=self.device) * self.dt
        self.t_test = t_test
        if norm_flag == 'zscore':
            y_test_norm, _, _, _ = _utils.normalize_data(
                y_test,
                norm_type=norm_flag,
                normalize=True,
                data_mean=self.x_min,
                data_std=self.x_max,
            )
            u_test_norm, _, _, _ = _utils.normalize_data(
                u_test,
                norm_type=norm_flag,
                normalize=True,
                data_mean=self.u_min,
                data_std=self.u_max,
            )
        else:
            y_test_norm, _, _, _ = _utils.normalize_data(
                y_test,
                norm_type=norm_flag,
                normalize=True,
                data_min=self.x_min,
                data_max=self.x_max,
            )
            u_test_norm, _, _, _ = _utils.normalize_data(
                u_test,
                norm_type=norm_flag,
                normalize=True,
                data_min=self.u_min,
                data_max=self.u_max,
            )


        if self.states_available is True:
            x_dot_test = torch.zeros_like(y_test,device=self.device)
            x_dot_test[1:-1] = (y_test[2:] - y_test[:-2]) / (2 * self.dt)
            x_dot_test[0] = (y_test[1] - y_test[0]) / self.dt
            x_dot_test[-1] = (y_test[-1] - y_test[-2]) / self.dt
            if norm_flag == 'zscore':
                x_dot_test_norm, _, _, _ = _utils.normalize_data(
                    x_dot_test,
                    norm_type=norm_flag,
                    normalize=True,
                    data_mean=self.x_dot_min,
                    data_std=self.x_dot_max,
                )
            else:
                x_dot_test_norm, _, _, _ = _utils.normalize_data(
                    x_dot_test,
                    norm_type=norm_flag,
                    normalize=True,
                    data_min=self.x_dot_min,
                    data_max=self.x_dot_max,
                )
            X_test_norm = torch.cat((y_test_norm, x_dot_test_norm), dim=-1)
            self.X_test_norm = X_test_norm
            self.X_dim = X_test_norm.size()[1]


        self.u_train_norm, self.y_train_norm =  u_train_norm, y_train_norm
        self.u_test_norm, self.y_test_norm = u_test_norm, y_test_norm
        self.u_dim, self.y_dim  = u_train_norm.size()[1], y_train_norm.size()[1]

    def _estimate_silverbox_euler_matrices(self):
        """
        Estimates c and k directly from normalized training data using least squares,
        then constructs physically interpretable Forward Euler matrices:
        
        A = [[ 1.0,           dt           ],
             [ -dt*(k/m),   1.0 - dt*(c/m) ]]
        B = [[ 0.0 ],
             [ dt/m ]]
        """
        y = self.y_train_norm[:, 0].detach().cpu()
        u = self.u_train_norm[:, 0].detach().cpu()
        dt = self.dt
        m = 1.0  # Mass parameter scaled to 1.0

        n = min(len(y), len(u))
        if n < 4:
            raise ValueError("Not enough training samples to estimate Euler parameters.")

        y = y[:n]
        u = u[:n]

        # 1. Target: y[k] for k = 2 ... n-1
        target = y[2:].unsqueeze(1)

        # 2. Regressors: y[k-1], y[k-2], u[k-1]
        phi = torch.stack((y[1:-1], y[:-2], u[1:-1]), dim=1)

        # 3. Fit discrete linear model: y[k] = a1*y[k-1] + a2*y[k-2] + b1*u[k-1]
        theta = torch.linalg.lstsq(phi, target).solution.squeeze()
        a1 = theta[0].item()
        a2 = theta[1].item()
        b1 = theta[2].item()

        # 4. Map discrete characteristic terms (z^2 - a1*z - a2 = 0) to continuous Euler terms
        # a1 = 2 - c*dt/m  -->  c = m * (2 - a1) / dt
        # a2 = (c*dt/m) - 1 - (k*dt^2/m)  -->  k = m * (1 - a1 - a2) / (dt^2)
        c_est = m * (2.0 - a1) / dt
        k_est = m * (1.0 - a1 - a2) / (dt ** 2)

        # Ensure parameters remain positive for stability
        c_est = max(c_est, 1e-4)
        k_est = max(k_est, 1e-4)

        # Store estimated physical properties on the instance
        self.c_est = c_est
        self.k_est = k_est
        self.m_est = m

        print(f"[Silverbox Euler Estimation]")
        print(f"  --> Estimated Damping (c): {c_est:.4f}")
        print(f"  --> Estimated Stiffness (k): {k_est:.4f}")

        # 5. Build explicit Euler matrices
        A = torch.tensor([
            [1.0, dt],
            [-dt * (k_est / m), 1.0 - dt * (c_est / m)]
        ], dtype=torch.float32, device=self.device)


        print(b1, dt / m)

        B = torch.tensor([
            [0.0],
            [dt / m]
        ], dtype=torch.float32, device=self.device)

        C = torch.tensor([[1.0, 0.0]], dtype=torch.float32, device=self.device)
        D = torch.tensor([[0.0]], dtype=torch.float32, device=self.device)

        return A, B, C, D

    def estimate_AB_matrices(self):
        """
        Estimates structured discrete state-space matrices A and B using native PyTorch.
        
        Parameters:
            u_train_norm: 1D PyTorch Tensor u(k)
            y_train_norm: 1D PyTorch Tensor y(k) (position state x_0)
            Ts: Sampling time (float or Tensor)
            
        Returns:
            A: (2, 2) PyTorch Tensor [[1.0, Ts], [a21, a22]]
            B: (2, 1) PyTorch Tensor [[0.0], [b2]]
        """
        # Ensure 1D Tensors on the same device and dtype
        u = self.u_train_norm.squeeze()
        y = self.y_train_norm.squeeze()
        device = y.device
        dtype = y.dtype

        # 1. Reconstruct velocity state x1(k) via finite difference
        x0 = y
        x1 = torch.zeros_like(y)
        x1[:-1] = (y[1:] - y[:-1]) / self.dt
        x1[-1] = x1[-2]  # Boundary condition

        # 2. Target for x1(k+1): x1(k+1) = a21*x0(k) + a22*x1(k) + b2*u(k)
        target = x1[1:].unsqueeze(1)  # Shape: (N-1, 1)
        
        # 3. Form Regressor Matrix Phi: [x0(k), x1(k), u(k)]
        Phi = torch.stack([x0[:-1], x1[:-1], u[:-1]], dim=1)  # Shape: (N-1, 3)
        
        # 4. Solve Linear Least Squares in PyTorch: Phi * theta = target
        solution = torch.linalg.lstsq(Phi, target)
        theta = solution.solution.squeeze(1)  # Shape: (3,)
        
        a21, a22, b2 = theta[0], theta[1], theta[2]
        
        # 5. Assemble state-space Tensors
        A = torch.tensor([[1.0, float(self.dt)],
                            [a21, a22]], device=device, dtype=dtype)
        
        B = torch.tensor([[0.0],
                            [b2]], device=device, dtype=dtype)

        C = torch.tensor([[1.0, 0.0]], device=device, dtype=dtype)
        D = torch.tensor([[0.0]], device=device, dtype=dtype)

        print(f"eigenvalues of A: {torch.linalg.eigvals(A)}")
        print(f"Estimated A matrix:\n{A}")
        print(f"Estimated B matrix:\n{B}")
        print(f"Estimated C matrix:\n{C}")
        print(f"Estimated D matrix:\n{D}")
        return A, B, C, D
    
    def _init_lin_matrices(self):
        
        def np_to_tensor(array):
            return torch.tensor(array, dtype=torch.float32, device=self.device)
        
        
        if self.test_case=='Silverbox':
            # Define system parameters
            # m = 1
            # c = 0.1
            # k = 1
            # m = 1.0
            # c = 326.3679
            # k = 175539.3280
            # self.A_init = torch.tensor([[1.0, self.dt],
            #                   [-self.dt*(k/m), 1.0 - (c/m) * self.dt]], dtype=torch.float32, device=self.device)
            
            # self.B_init = torch.tensor([[0.0],
            #                   [self.dt]], dtype=torch.float32, device=self.device)
            # self.C_init = torch.tensor([[1.0, 0.0]], dtype=torch.float32, device=self.device)
            # self.D_init = torch.tensor([[0.0]], dtype=torch.float32, device=self.device)
            # self.A_init, self.B_init, self.C_init, self.D_init = self.estimate_AB_matrices()
            self.A_init = torch.tensor([[1, 0], 
                                        [0, 1]], dtype=torch.float32, device=self.device
                                        )
            self.B_init = torch.tensor([[0],
                                        [1]], dtype=torch.float32, device=self.device
                                        )
            self.C_init = torch.tensor([[1, 0]], dtype=torch.float32, device=self.device
                                        )
            self.D_init = torch.tensor([[0]], dtype=torch.float32, device=self.device
                                        )
            print(f"eigenvalues of A: {torch.linalg.eigvals(self.A_init)}")
            print(f"A matrix:\n{self.A_init}")
            print(f"B matrix:\n{self.B_init}")
            print(f"C matrix:\n{self.C_init}")
            print(f"D matrix:\n{self.D_init}")

        elif self.test_case == 'Wiener-Hammerstein':
            if self.init_matrices_flag == 'filters':
                fs = 51200
                def create_chebyshev1_filter():
                    b, a = signal.cheby1(N=3, rp=0.5, Wn=4400, btype='low', fs=fs)
                    return signal.tf2ss(b, a)
                def create_chebyshev2_filter():
                    b, a = signal.cheby2(N=3, rs=40, Wn=5000, btype='low', fs=fs)
                    return signal.tf2ss(b, a)

                A1, B1, C1, D1 = create_chebyshev1_filter()
                A2, B2, C2, D2 = create_chebyshev2_filter()
                self.A1_init = np_to_tensor(A1)
                self.B1_init = np_to_tensor(B1)
                self.C1_init = np_to_tensor(C1)
                self.D1_init = torch.tensor(np.array(0).reshape(-1,1), dtype=torch.float32, device=self.device)
                self.A2_init = np_to_tensor(A2)
                self.B2_init = np_to_tensor(B2)
                self.C2_init = np_to_tensor(C2)
                self.D2_init = torch.tensor(np.array(0).reshape(-1,1), dtype=torch.float32, device=self.device)
                
            elif self.init_matrices_flag == 'matlab':
                import scipy.io
                mat_path = 'matlab22.mat' 
                mat = scipy.io.loadmat(mat_path)
                A=mat['A_train']
                B=mat['B_train']
                C=mat['C_train']
                
                self.A_init = np_to_tensor(A)
                self.B_init = np_to_tensor(B)
                self.C_init = np_to_tensor(C)
                self.D_init = torch.tensor(np.array(0).reshape(-1,1), dtype=torch.float32, device=self.device)
        
        elif self.test_case == 'Luca-Airfoil-CFD':
            if self.init_matrices_flag == 'matlab_1':
                if self.norm_flag == 'minmax':
                    mat_path = 'TimeSeries_Exp_CFD/matlab_norm_foil_1state.mat'
                elif self.norm_flag == 'nothing':
                    mat_path = 'TimeSeries_Exp_CFD/matlab_NOTnorm_foil_1state.mat'

            elif self.init_matrices_flag == 'matlab_4':
                if self.norm_flag == 'minmax':
                    mat_path = 'TimeSeries_Exp_CFD/matlab_norm_foil_4state.mat'
                elif self.norm_flag == 'nothing':
                    mat_path = 'TimeSeries_Exp_CFD/matlab_norm_foil_4state.mat'

            elif self.init_matrices_flag == 'ones':
                return
            import scipy.io
            mat = scipy.io.loadmat(mat_path)
            A=mat['A_train']
            B=mat['B_train']
            C=mat['C_train']
                
            self.A_init = np_to_tensor(A)
            self.B_init = np_to_tensor(B)
            self.C_init = np_to_tensor(C)
            self.D_init = torch.tensor(np.array(0).reshape(-1,1), dtype=torch.float32, device=self.device)
            
            
        elif self.test_case == 'Luca-Airfoil-Exp':
            if self.init_matrices_flag == 'matlab_1':
                if self.norm_flag == 'minmax':
                    mat_path = 'TimeSeries_Exp_CFD/matlab_norm_foil_1state_EXP.mat'
                elif self.norm_flag == 'nothing':
                    mat_path = None
                    
            if self.init_matrices_flag == 'matlab_2':
                if self.norm_flag == 'minmax':
                    mat_path = '/Users/cruz/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/Python_scripts/ss-kan-paper/TimeSeries_Exp_CFD/matlab_norm_foil_2state_EXP.mat'

            elif self.init_matrices_flag == 'ones':
                return
            
            import scipy.io
            mat = scipy.io.loadmat(mat_path)
            A=mat['A_train']
            B=mat['B_train']
            C=mat['C_train']
                
            self.A_init = np_to_tensor(A)
            self.B_init = np_to_tensor(B)
            self.C_init = np_to_tensor(C)
            self.D_init = torch.tensor(np.array(0).reshape(-1,1), dtype=torch.float32, device=self.device)
            #self.B_init = torch.tensor(np.array(1).reshape(-1,1), dtype=torch.float32, device=self.device)
            #self.C_init = torch.tensor(np.array(1).reshape(-1,1), dtype=torch.float32, device=self.device)

        elif self.test_case == 'Luca-Airfoil-Exp-pitchRate':

            self.A_init = torch.tensor(np.array(1).reshape(-1,1), dtype=torch.float32, device=self.device)
            self.B_init = torch.tensor(np.array([0, 0]).reshape(-1,1).T, dtype=torch.float32, device=self.device)
            self.C_init = torch.tensor(np.array(-125).reshape(-1,1), dtype=torch.float32, device=self.device)
            self.C_init = torch.tensor(np.array(1).reshape(-1,1), dtype=torch.float32, device=self.device)
            self.D_init = torch.tensor(np.array([0, 0]).reshape(-1,1).T, dtype=torch.float32, device=self.device)

        elif self.test_case == 'Luca-Airfoil-CFD-pitchRate':

            self.A_init = torch.tensor(np.array(1).reshape(-1,1), dtype=torch.float32, device=self.device)
            self.B_init = torch.tensor(np.array([0, 0]).reshape(-1,1).T, dtype=torch.float32, device=self.device)
            self.C_init = torch.tensor(np.array(1).reshape(-1,1), dtype=torch.float32, device=self.device)
            self.D_init = torch.tensor(np.array([0, 0]).reshape(-1,1).T, dtype=torch.float32, device=self.device)

        return
    
    