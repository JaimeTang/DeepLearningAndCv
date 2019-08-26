# It's empty. Surprise!
# Please complete this by yourself.
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(3, 3, 3),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool2d(kernel_size=2),
                                  nn.Conv2d(3, 6, 3),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool2d(kernel_size=2))

        self.dense = nn.Sequential(nn.Linear(6 * 123 * 123, 150),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout2d(p=0.2))

        self.fc1 = nn.Sequential(nn.Linear(150, 2),
                                 nn.Sigmoid())
        self.fc2 = nn.Sequential(nn.Linear(150, 3),
                                 nn.Softmax())



    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 6 * 123 * 123)
        x = self.dense(x)

        output_classes = self.fc1(x)
        output_species = self.fc2(x)

        return output_classes, output_species