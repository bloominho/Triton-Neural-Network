import torch
import os
from ResNet18 import ResNet18
import util

if __name__ == "__main__":
    model = ResNet18(num_classes=10)
    pretrained_save_path = "data/resnet18_cifar10.pth"
    result_save_path = "result/accuracy.txt"
    
    if os.path.exists(pretrained_save_path):
        print("Loading pretrained model...")
        model.load_state_dict(torch.load(pretrained_save_path))
    else:
        assert False, "Pretrained model does not exist"
    
    print("Start Loading.")
    trainloader, testloader = util.get_dataloader()

    print("Start Training.")
    util.train_model(model, trainloader, testloader)