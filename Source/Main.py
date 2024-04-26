from Model_20 import Autoencoder
from utils_V2 import Reconstructed_writer

import numpy as np
from sklearn.model_selection import train_test_split
#from torch.utils.data import random_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F 

import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import time

# to control randomization, a seed value is assigned  to make code repeatable
my_seed = 24
def set_seed (my_seed = 24):
  np.random.seed(my_seed)
  torch.manual_seed(my_seed)
  torch.cuda.manual_seed(my_seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

set_seed(my_seed=my_seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print(device)

data = np.load("flowfield_hom.npy")
minibatch = 5 # # how many samples per batch to load
print(data.shape)
if minibatch == None:
  minibatch = data.shape[0]

# uploading the data from numpy file

flow_cond = np.load("flowcon_hom.npy")
x_coordinates = np.load("x_coordinate_hom.npy")
y_coordinates = np.load("y_coordinate_hom.npy")

print(data.shape)
#data = data.to(device)
print(x_coordinates.shape)
#to split randomly
print(flow_cond.shape)

# normalizing the flow field
flowfield_mean = np.mean(data, axis=0) # cell-based mean values
flowfield_std = np.std(data, axis=0) # cell-based std values
print(flowfield_mean.shape)

normalized_data = (data - flowfield_mean) / flowfield_std

# splitting train and validation data (coordinates, AoA whole together shuffled)
train_data, val_data, train_xcoord, test_xcoord, train_ycoord, test_ycoord, train_flowcon, val_flowcon = train_test_split(normalized_data, x_coordinates, y_coordinates, flow_cond, test_size=0.2)
print(train_data[1].shape)

# make numpy to tensor
torch.set_default_dtype(torch.float64)

train_loader = DataLoader(torch.tensor(train_data), batch_size=minibatch, shuffle=False)
val_loader = DataLoader(torch.tensor(val_data), batch_size=val_data.shape[0], shuffle=False)

# Root mean square error
criterion = nn.MSELoss()
def RMSELoss(recon_x,x):
    return torch.sqrt(criterion(recon_x,x))

# number of epochs
n_epochs = 500

lr_step_size = 500
lr_gamma = 0.2
set_seed (my_seed)
print(torch.cuda.get_device_name(0)) 
model = Autoencoder().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f'Toplam parametre sayısı: {total_params}') # to see total parameters


optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = lr_scheduler.StepLR(optimizer, lr_step_size, gamma=lr_gamma)
start = time.time()
min_val_loss = np.Inf


#val_losses = []
loss_his = []

for epoch in tqdm(range(n_epochs)):
    #monitor training loss and validation loss
    train_loss = 0.0
    
    # Empty the outputs list at the beginning of each epoch
    outputs = []
    ###################
    # train the model #
    ###################
    for idx, (images) in enumerate(train_loader):
        images = images.to(device)
        recon_images = model(images)
        loss = criterion(recon_images, images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        outputs.append(recon_images)
    loss_his.append(loss.item())
    #scheduler.step()
          
    

    if epoch % 10 == 0:# and epoch != 0:
      
      to_print = "Epoch[{}/{}] Time: {:.0f} Loss: {:.6f}".format(epoch+1, 
                              n_epochs, time.time()-start, loss.item())
      print(to_print)    


# concatenate outputs list to get the final output tensor
final_output = torch.cat(outputs, dim=0)
print(final_output[1,0,:,:].shape)

with torch.no_grad():
    model.eval()
    for images in val_loader:
      images = images.to(device)
      recon_images_val = model(images)
      loss_val_p = criterion(images[:,0], recon_images_val[:,0])
      loss_val_u = criterion(images[:,1], recon_images_val[:,1])
      loss_val_v = criterion(images[:,2], recon_images_val[:,2])
      print(loss_val_p)
      print(loss_val_u)
      print(loss_val_v)

# concatenate outputs list to get the final output tensor
final_output_val = torch.cat((recon_images_val,), dim=0)  



# changing normalized data back to orginal distribution for training dataset
reconstructed_data = torch.zeros(len(train_data),3,150,498)
for i in range(3):
    for k in range(len(train_data)):
        reconstructed_data[k,i,:,:] = final_output[k,i,:,:]* torch.tensor(flowfield_std).to(device)[i] + torch.tensor(flowfield_mean).to(device)[i]
        
# Origininal data
original_data = torch.zeros(len(train_data),3,150,498)
for i in range(3):
    for k in range(len(train_data)):
        original_data[k,i,:,:] = torch.tensor(train_data[k,i,:,:]).to(device)* torch.tensor(flowfield_std).to(device)[i] + torch.tensor(flowfield_mean).to(device)[i]
    
# error data
error_data = abs(original_data - reconstructed_data) / original_data

# changing normalized data back to orginal distribution for validation dataset
reconstructed_val = torch.zeros(len(val_data),3,150,498)
for i in range(3):
    for k in range(len(val_data)):
        reconstructed_val[k,i,:,:] = final_output_val[k,i,:,:]* torch.tensor(flowfield_std).to(device)[i] + torch.tensor(flowfield_mean).to(device)[i]

# location of write files
phat_file = r"C:\Users\Abdullah\Desktop\ASDL\Optimization_Team\Toy_Problem\Model\output\v4\P"
uhat_file = r"C:\Users\Abdullah\Desktop\ASDL\Optimization_Team\Toy_Problem\Model\output\v4\u"
vhat_file = r"C:\Users\Abdullah\Desktop\ASDL\Optimization_Team\Toy_Problem\Model\output\v4\v"

#error file locations
pe_file = r"C:\Users\Abdullah\Desktop\ASDL\Optimization_Team\Toy_Problem\Model\output\v4\error\P_error"
ue_file = r"C:\Users\Abdullah\Desktop\ASDL\Optimization_Team\Toy_Problem\Model\output\v4\error\u_error"
ve_file = r"C:\Users\Abdullah\Desktop\ASDL\Optimization_Team\Toy_Problem\Model\output\v4\error\v_error"

# writing constructed data to .dat files / train data
train_write = Reconstructed_writer(dataset=reconstructed_data, flow_cond = flow_cond, flowcon = train_flowcon, phat_file=phat_file, uhat_file=uhat_file, vhat_file=vhat_file, num_cases=len(reconstructed_data), num_rows=150, num_cols=498, num_channels=3) # num_channels-1 due to AoA as channel
export_train = train_write.write_dataset()

# writing error to .dat files / error data
error_write = Reconstructed_writer(dataset=error_data, flow_cond = flow_cond, flowcon = train_flowcon, phat_file=pe_file, uhat_file=ue_file, vhat_file=ve_file, num_cases=len(reconstructed_data), num_rows=150, num_cols=498, num_channels=3) # num_channels-1 due to AoA as channel
export_error = error_write.write_dataset()

# writing constructed data to .dat files / validation data
val_write = Reconstructed_writer(dataset=reconstructed_val, flow_cond = flow_cond, flowcon = val_flowcon, phat_file=phat_file, uhat_file=uhat_file, vhat_file=vhat_file, num_cases=len(reconstructed_val), num_rows=150, num_cols=498, num_channels=3) # num_channels-1 due to AoA as channel
export_val = val_write.write_dataset()

# Save the model
#torch.save(model.state_dict(), r'C:\Users\Abdullah\Desktop\ASDL\Optimization_Team\Toy Problem\Model\conv_autoencoder.pth')

# Load the state of the model from the conv_autoencoder file
# model.load_state_dict(torch.load(r'C:\Users\Abdullah\Desktop\ASDL\Optimization_Team\Toy Problem\Model\conv_autoencoder.pth'))

 # Plotting the training and validation losses
 
#plt.plot(range(1, n_epochs+1), train_losses, label='Training Loss')
#plt.plot(range(1, n_epochs+1), val_losses, label='Validation Loss')
#plt.xlabel('Epoch')
#plt.ylabel('Loss')
#plt.title('Training and Validation Losses')
#plt.legend()
#plt.show()
