from torch import nn

myawesomemodel = nn.Sequential(
    nn.Conv2d(1, 32, 3),
    nn.LeakyReLU(),
    nn.Conv2d(32, 64, 3),
    nn.LeakyReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(64 * 12 * 12, 10)
    
)