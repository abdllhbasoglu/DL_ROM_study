import torch
import numpy as np
import os
import copy
torch.set_default_dtype(torch.float64)

class DatasetLoader:
    def __init__(self, p_file, u_file, v_file, AoA_file, num_cases, num_rows, num_cols, num_channels):
        self.p_file = p_file
        self.u_file = u_file
        self.v_file = v_file
        self.AoA_file = AoA_file
        self.num_cases = num_cases
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_channels = num_channels
        self.dataset = None

    def load_dataset(self):
        self.dataset = torch.zeros((self.num_cases, self.num_channels, self.num_rows, self.num_cols))
        Angle_of_attack = os.path.join(self.AoA_file, "AoA.dat")
        AoA_data = np.loadtxt(Angle_of_attack, dtype=np.float64)
        for i in range(self.num_cases):

            # File name "1_output_p.dat", "1_output_u.dat", "1_output_v.dat" and go on
            pressure_file = os.path.join(self.p_file, f"{i+1}_output_p.dat")
            u_velocity_file = os.path.join(self.u_file, f"{i+1}_output_u.dat")
            v_velocity_file = os.path.join(self.v_file, f"{i+1}_output_v.dat")
            
            
            # ASCII format to matrix of ixj with a precision 9
            pressure_data = np.loadtxt(pressure_file, dtype=np.float64).reshape(self.num_rows, self.num_cols)
            u_velocity_data = np.loadtxt(u_velocity_file, dtype=np.float64).reshape(self.num_rows, self.num_cols)
            v_velocity_data = np.loadtxt(v_velocity_file, dtype=np.float64).reshape(self.num_rows, self.num_cols) 
            
            # AoA to 150x498 size
            AoA_Matrix = np.full((self.num_rows, self.num_cols), AoA_data[i])
                                 
            # numpy to torch tensor transformation
            self.dataset[i, 0, :, :] = torch.tensor(pressure_data)
            self.dataset[i, 1, :, :] = torch.tensor(u_velocity_data)
            self.dataset[i, 2, :, :] = torch.tensor(v_velocity_data)
            self.dataset[i, 3, :, :] = torch.tensor(AoA_Matrix)

            #self.dataset[i, 0, :, :] = torch.from_numpy(pressure_data)
            #self.dataset[i, 1, :, :] = torch.from_numpy(u_velocity_data)
            #self.dataset[i, 2, :, :] = torch.from_numpy(v_velocity_data)

        return self.dataset

class Normalize:
    def __init__(self, org_data, num_channels):
        self.org_data= org_data
        self.num_channels = num_channels
        self.mean_t = None
        self.std_t = None
        self.normalized_data = None

    def normalize(self):
        self.normalized_data = copy.deepcopy(self.org_data)
        self.mean_t = torch.zeros(len(self.org_data), self.num_channels)
        self.std_t = torch.zeros(len(self.org_data), self.num_channels)

        for i in range(self.num_channels):
            for j in range(len(self.org_data)):
                self.mean_t[j,i] = self.org_data[j][i,:,:].mean()
                self.std_t[j,i] = self.org_data[j][i,:,:].std()
                self.normalized_data[j][i,:,:] = (self.org_data[j][i,:,:] - self.mean_t[j,i]) / self.std_t[j,i]
                self.normalized_data[j][3,:,:] = self.org_data[j][3,:,:] # angle of attack channel kept same
                
        return self.normalized_data

class Denormalize:
    def __init__(self, normalized_data, num_channels, mean, std):
        self.normalized_data= normalized_data
        self.num_channels = num_channels
        self.mean = mean
        self.std = std
        self.denormalized_data = None

    def denormalize(self):
        self.denormalized_data = torch.zeros_like(self.normalized_data)

        for i in range(self.num_channels):
            for j in range(len(self.normalized_data)):
                self.denormalized_data[j][i,:,:] = (self.normalized_data[j][i,:,:] * self.std[j,i] ) + self.mean[j,i] 
                

        return self.denormalized_data

class Reconstructed_writer:
    def __init__(self, dataset, flow_cond, flowcon, phat_file, uhat_file, vhat_file, num_cases, num_rows, num_cols, num_channels):
        self.dataset = dataset
        self.flow_cond = flow_cond
        self.flowcon = flowcon
        self.phat_file = phat_file
        self.uhat_file = uhat_file
        self.vhat_file = vhat_file
        self.num_cases = num_cases
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_channels = num_channels


    def write_dataset(self):
        for i in range(self.num_cases):

            indices = np.where(self.flow_cond == self.flowcon[i])[0]
            # File name "1_output_p.dat", "1_output_u.dat", "1_output_v.dat" and go on
            pressure_file = os.path.join(self.phat_file, f"{indices[0]+1}_output_p.dat")
            u_velocity_file = os.path.join(self.uhat_file, f"{indices[0]+1}_output_u.dat")
            v_velocity_file = os.path.join(self.vhat_file, f"{indices[0]+1}_output_v.dat")

            #CPU
            #pressure_data = self.dataset[i,0,:,:].detach().numpy().reshape(self.num_rows*self.num_cols, 1)
            #u_velocity_data = self.dataset[i,1,:,:].detach().numpy().reshape(self.num_rows*self.num_cols, 1)
            #v_velocity_data = self.dataset[i,2,:,:].detach().numpy().reshape(self.num_rows*self.num_cols, 1)

            #CUDA
            pressure_data = self.dataset[i,0,:,:].cpu().detach().numpy().reshape(self.num_rows*self.num_cols, 1)
            u_velocity_data = self.dataset[i,1,:,:].cpu().detach().numpy().reshape(self.num_rows*self.num_cols, 1)
            v_velocity_data = self.dataset[i,2,:,:].cpu().detach().numpy().reshape(self.num_rows*self.num_cols, 1)

            np.savetxt(pressure_file, pressure_data, fmt='%.9e')
            np.savetxt(u_velocity_file, u_velocity_data, fmt='%.9e')
            np.savetxt(v_velocity_file, v_velocity_data, fmt='%.9e')
