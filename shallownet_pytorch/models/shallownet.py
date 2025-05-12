# import the necessary packages
import torch.nn as nn
import torch.nn.functional as F

class ShallowNet(nn.Module):
    def __init__(self, width, height, depth, classes):
        super(ShallowNet, self).__init__()

        # Definindo a camada Conv2D com padding para manter o tamanho da imagem
        self.conv = nn.Conv2d(in_channels=depth, out_channels=32, kernel_size=3, padding=1)

        # Flatten será feito no método forward usando .view()

        # Camada densa (fully connected)
        # Como padding='same' e stride=1, a saída tem a mesma altura e largura da entrada
        self.fc = nn.Linear(32 * height * width, classes)

    def forward(self, x):
        # x tem shape (batch_size, depth, height, width)
        x = self.conv(x)
        x = F.relu(x)

        # achata para (batch_size, features)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        #x = F.softmax(x, dim=1) #- se não usar crossentropy descomentar 
        return x