from torch import nn

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
