import torch
import torch.nn as nn
from torchviz import make_dot
from torchsummary import summary

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder kısmı
        self.encoder = nn.Sequential(
            #input channel arrange
            self.conv_block(3, 16, 1, padding=0),
            
            #1
            nn.MaxPool2d(2, padding=(1, 1)),
            self.conv_block(16, 32, 3),
            
            #2
            nn.MaxPool2d(2, padding=(0, 1)),
            self.conv_block(32, 64, 3),
            
            #3
            nn.MaxPool2d(2, padding=(1, 1)),
            self.conv_block(64, 128, 3),
            
            #4
            nn.MaxPool2d(2, padding=(0, 0)),
            self.conv_block(128, 128, 3),
            
            #5
            nn.MaxPool2d(2, padding=(1, 0)),
            self.conv_block(128, 128, 3),
            
            #6
            nn.MaxPool2d(2, padding=(1, 0)),
            self.conv_block(128, 128, 3),
            
            #7
            nn.MaxPool2d(2, padding=(0, 0)),
            self.conv_block(128, 128, 3),
            
            #Flatten and FCN
            nn.Flatten(),
            nn.Linear(128 * 2 * 4, 32)
        
        )

        # Decoder kısmı
        self.decoder = nn.Sequential(
            nn.Linear(32, 128 * 2 * 4),
            nn.Unflatten(1, (128, 2, 4)),
            
            #1
            nn.Upsample(size=(4,8), mode='bicubic'),
            self.convTranspose_block(128, 128, 3),
                        
            #2
            nn.Upsample(size=(6,16), mode='bicubic'),
            self.convTranspose_block(128, 128, 3),
                                    
            #3
            nn.Upsample(size=(10,32), mode='bicubic'),
            self.convTranspose_block(128, 128, 3),
                        
            #4
            nn.Upsample(size=(20,64), mode='bicubic'),
            self.convTranspose_block(128, 128, 3),
                 
            #5
            nn.Upsample(size=(38,126), mode='bicubic'),
            self.convTranspose_block(128, 64, 3),
                        
            #6
            nn.Upsample(size=(76,250), mode='bicubic'),
            self.convTranspose_block(64, 32, 3),
                                   
            #7
            nn.Upsample(size=(150,498), mode='bicubic'),
            self.convTranspose_block(32, 16, 3),
                                   
            #output
            nn.Conv2d(16, 3, 1, stride=1)
        )

    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed

    def encode(self, x):
        output = self.encoder(x)
        return output

    def decode(self, x):
        output = self.decoder(x)
        return output

    def conv_block(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
    def convTranspose_block(self, in_channels, out_channels, kernel_size, stride=1, padding=1, Out_P=0):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, Out_P),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


################################ Testing Model ######################################
# Autoencoder'i oluştur
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#autoencoder = Autoencoder().to(device=device)
#print(autoencoder)

#summary(autoencoder, (3, 150, 498), device='cuda')
# Giriş verisi
#input_data = torch.randn(3, 3, 150, 498).to(device)

# Autoencoder üzerinden geçiş
#output = autoencoder(input_data)

# Verify model output
#print("Output shape:", output.shape)



# to make a graph of model architecture
# Model grafiğini oluştur
#dot = make_dot(output, params=dict(autoencoder.named_parameters()))
#dot.render("CNN model", format="png")
# Display the graph directly
#dot.view()


