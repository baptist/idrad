import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BaseNet(nn.Module):

    def __init__(self, input_dim, output_dim, dynamic_pooling=True):

        super(BaseNet, self).__init__()

        doppler_pooling = np.zeros((4,), dtype=np.int32)
        time_pooling = np.zeros((4,), dtype=np.int32)

        if dynamic_pooling:
            doppler_pooling[:] = int((float(input_dim[2]) / 5) ** .25)
            res = ((float(input_dim[2]) / 5) ** .25) - doppler_pooling[0]
            for i in range(int(round(res * 4))):
                doppler_pooling[i] += 1
        else:
            doppler_pooling[:] = 2;
            c = 0
            while input_dim[2] < np.prod(doppler_pooling):
                doppler_pooling[-(c % 4) - 1] -= 1;
                c += 1

        if dynamic_pooling:
            time_pooling[:] = int((float(input_dim[1]) / 5) ** .25)
            res = ((float(input_dim[1]) / 5) ** .25) - time_pooling[0]
            for i in range(int(round(res * 4))):
                time_pooling[i] += 1
        else:
            time_pooling[:] = 2;
            c = 0
            while input_dim[1] < np.prod(time_pooling):
                time_pooling[-(c % 4) - 1] -= 1;
                c += 1

        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(in_channels=input_dim[0], out_channels=8, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.MaxPool2d((time_pooling[0], doppler_pooling[0])),
            # Conv2
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.MaxPool2d((time_pooling[1], doppler_pooling[1])),
            # Conv3
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.MaxPool2d((time_pooling[2], doppler_pooling[2])),
            # Conv4
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.MaxPool2d((time_pooling[3], doppler_pooling[3]))
        )  # --- Conv sequential ends ---

        self.classifier = nn.Sequential(
            nn.Linear(64 * 5 * 5, 128),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, output_dim)
        )  # --- Linear sequential ends ---



    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 64*5*5)
        return self.classifier(x)



class OrigNet(nn.Module):

    def __init__(self, input_dim, output_dim):

        super(OrigNet, self).__init__()

        # Define all layers containing learnable weights.
        self.conv1 = nn.Conv2d(in_channels=input_dim[0], out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.c0_dropout = nn.Dropout2d(0.)
        self.c1_dropout = nn.Dropout2d(0.2)
        self.c2_dropout = nn.Dropout2d(0.4)
        self.dropout = nn.Dropout(0.5)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 2 * 12, 128)
        self.fc2 = nn.Linear(128, output_dim)



    def forward(self, x):
        x = self.pool(self.c0_dropout(F.elu(self.conv1(x))))
        x = self.pool(self.c1_dropout(F.elu(self.conv2(x))))
        x = self.pool(self.c1_dropout(F.elu(self.conv3(x))))
        x = self.pool(self.c2_dropout(F.elu(self.conv4(x))))

        x = x.view(-1, 64 * 2 * 12)

        x = self.dropout(F.elu(self.fc1(x)))
        # x = F.softmax(self.fc2(x), dim=1)
        x = self.fc2(x)

        return x