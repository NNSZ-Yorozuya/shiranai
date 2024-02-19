# 部署，用于单张图片识别
import torch
import os
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def eval_model(
    model: nn.Module,
    image: torch.Tensor,
):
    model.eval()
    images = image.unsqueeze(0)
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    return predicted[0]

def eval():
    from data import get_test_loader
    from nets import SimpleClassifier, CNN2
    model = CNN2()
    sd = torch.load(os.path.join('trained', 'cnn2-adam-0.001-0.9-30-0.1-8.ckpt'))
    model.load_state_dict(sd)
    model = model.to(device)
    
    eval_model(model, )


if __name__ == "__main__":
    eval()