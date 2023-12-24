from torch import nn
import torch.nn.functional as F

# 输入张量一定是45 * 70 * 3的，需要改可以去utils.py里面改resize_image函数

# 定义模型
class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(45*70*3, 37)  # 45*70 pixels * 3 channels RGB

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x

class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 5) # 41 * 66 * 10
        self.pool = nn.MaxPool2d(2, 2) # 20 * 33 * 10
        self.conv2 = nn.Conv2d(10, 26, 5) # 16 * 29 * 26
        self.fc1 = nn.Linear(8 * 14 * 26, 300)
        self.fc2 = nn.Linear(300, 124)
        self.fc3 = nn.Linear(124, 37)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 8 * 14 * 26)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
