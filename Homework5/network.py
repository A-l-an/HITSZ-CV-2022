from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        """
        Init network.

        The structure of network:
            * 2D convolution: output feature channel number = 6, kernel size = 5x5, stride = 1, no padding;
            * 2D max pooling: kernel size = 2x2, stride = 2;
            * 2D convolution: output feature channel number = 16, kernel size = 5x5, stride = 1, no padding;
            * 2D max pooling: kernel size = 2x2, stride = 2;
            * Fully-connected layer: output feature channel number = 120;
            * Fully-connected layer: output feature channel number = 84;
            * Fully-connected layer: output feature channel number = 10 (number of classes).

        Hint:
            1. for 2D convolution, you can use `torch.nn.Conv2d`
            2. for 2D max pooling, you can use `torch.nn.MaxPool2d`
            3. for fully connected layer, you can use `torch.nn.Linear`
            4. Before the first fully connected layer, you should have a tensor with shape (BatchSize, 16, 5, 5),
               later in `forward()` you can flatten it to shape `(BatchSize, 400)`, 
               so the `input_feature` of the first connected layer is 400.
        """
        super().__init__()
        ### YOUR CODE HERE
        layers = []
        in_channels  = 3
        out_channels_list = [6, 16]
        # Convolution & Pooling layers
        for out_channels in out_channels_list:
            layers += [nn.Conv2d(in_channels, out_channels, kernel_size=5,stride=1, padding=0)]
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)
        # Fully-connected layer
        self.fc = nn.Sequential(
                        nn.Linear(400, 120),
                        nn.Linear(120, 84) ,
                        nn.Linear(84, 10)
        )
        ### END YOUR CODE

    def forward(self, x):
        """
        Forwrad process.

        Hint:
            Before the first fully connected layer, you should have a tensor with shape (BatchSize, 16, 5, 5),
            you can flatten the tensor to shape `(BatchSize, 400)` here, you may find `torch.flatten` helpful.
        """
        ### YOUR CODE HERE
        # x = torch.flatten(x, 1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)   # size(0) is BatchSize
        x = self.fc(x)
        ### END YOUR CODE
        return x