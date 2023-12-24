from typing import Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # 实例化模型，并移动到设备上
# model = SimpleClassifier().to(device)

# 定义损失函数和优化器
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 假设你已经有了一个数据加载器data_loader
# 下面是训练模型的框架
def train_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.modules.loss._WeightedLoss,
    optimizer: optim.Optimizer,
    num_epochs: int,
    ckpt_path: Path = None,
    filename_prefix = 'model'
    ):
    ckpt_path = ckpt_path or Path()
    ckpt_path.mkdir(exist_ok=True, parents=True)
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for images, labels in data_loader:
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()  # Zero the parameter gradients
            loss.backward()  # Perform backward pass
            optimizer.step()  # Update parameters

            # Print statistics
            running_loss += loss.item()
        
        # Print epoch loss
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(data_loader):.4f}')
        # Save model checkpoint
        torch.save(model.state_dict(), ckpt_path / f'{filename_prefix}-{epoch+1}.ckpt')

# 训练模型
# train_model(model, data_loader, criterion, optimizer, num_epochs=5)

if __name__ == "__main__":
    from data import get_train_loader
    from nets import SimpleClassifier
    model = SimpleClassifier().to(device)
    data_loader = get_train_loader()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train_model(model, data_loader, criterion, optimizer, num_epochs=5,
        ckpt_path=Path(r'F:\shiranai\trained'), filename_prefix='simple-0.01-0.9')