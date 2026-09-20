import torch 
import torch.nn as nn


#-------------DEVICE------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#---------------MODEL BUILDING------------
class Encoder(nn.Module):
    def __init__(self):
        pass
    
    def forward(self):
        pass
    
class Decoder(nn.Module):
    def __init__(self):
        pass
    def forward(self):
        pass
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        pass
    def forward(self):
        pass

#------------------PARAMETERS--------------------

learning_rate = 0.01
epochs = 25
optimzer = None
criterion = nn.CrossEntropyLoss()

#-------------------TRAINING---------------------
for ep in range(epochs):
    pass



#------------------VALIDATION---------------------