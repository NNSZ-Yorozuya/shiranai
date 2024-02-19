from typing import Tuple
from pathlib import Path

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import DataLoader

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 使用预定义的初始化方法初始化权重
def init_weights(m):
    if type(m) in (nn.Linear, nn.Conv2d):
        init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)

# 假设你已经有了一个数据加载器data_loader
# 下面是训练模型的框架
def train_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.modules.loss._WeightedLoss,
    optimizer: optim.Optimizer,
    num_epochs: int,
    ckpt_path: Path = None,
    filename_prefix = 'model',
    resume: str = None
    ):
    ckpt_path = ckpt_path or Path()
    ckpt_path.mkdir(exist_ok=True, parents=True)
    if resume:
        sd = torch.load(resume)
        model.load_state_dict(sd)
    else:
        model.apply(init_weights)
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct_predictions = 0

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

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
        
        # Print epoch loss
        accuracy = correct_predictions / len(data_loader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(data_loader):.4f}, Accuracy: {accuracy:.4f}')
        # Save model checkpoint
        torch.save(model.state_dict(), ckpt_path / f'{filename_prefix}-{epoch+1}.ckpt')

def test_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.modules.loss._WeightedLoss,
):
    # 设置模型为评估模式
    model.eval()
    
    # 初始化损失和正确的预测数量
    total_loss = 0
    correct_predictions = 0
    
    # 不计算梯度
    with torch.no_grad():
        for images, labels in data_loader:
            # 将数据移动到配置的设备上
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播，计算预测结果
            outputs = model(images)
            
            # 计算损失
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
    
    # 计算平均损失和准确率
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / len(data_loader.dataset)
    
    print(f'Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    return avg_loss, accuracy

# 训练模型
# train_model(model, data_loader, criterion, optimizer, num_epochs=5)

def train():
    from data import get_train_loader
    from nets import SimpleClassifier, CNN2
    # 超参数
    learn_rate = 0.001
    momentum = 0.9
    epoch = 30
    weight_decay = 0.1
    #
    model = CNN2()
    model = model.to(device)
    data_loader = get_train_loader()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learn_rate,
        # momentum=momentum,
        weight_decay=weight_decay
    )
    train_model(model, data_loader, criterion, optimizer, num_epochs=epoch,
        ckpt_path=Path(r'trained'), filename_prefix=f'cnn2-adam-{learn_rate}-{momentum}-{epoch}-{weight_decay}')

def test():
    from data import get_test_loader
    from nets import SimpleClassifier, CNN2
    # model = SimpleClassifier()
    model = CNN2()
    # sd = torch.load(r'F:\shiranai\trained\simple-0.01-0.9-2.ckpt')
    for mdlpath in os.listdir('trained'):
        if mdlpath.endswith('.ckpt'):
            sd = torch.load(os.path.join('trained', mdlpath))
            model.load_state_dict(sd)
            model = model.to(device)
            data_loader = get_test_loader()
            criterion = nn.CrossEntropyLoss()
            print(mdlpath)
            test_model(model, data_loader, criterion)
    # sd = torch.load(r'F:\shiranai\trained\cnn2-0.01-0.9-5.ckpt')
    # model.load_state_dict(sd)
    # model = model.to(device)
    # data_loader = get_test_loader()
    # criterion = nn.CrossEntropyLoss()
    # test_model(model, data_loader, criterion)

if __name__ == "__main__":
    # train()
    test()