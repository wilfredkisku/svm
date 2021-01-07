import torchvision.models as models
import torch.nn
import torch

##############Getting the pretrained Model######################
resnet50 = models.resnet50(pretrained=True)
print(resnet50)

##############Stripping to the last layer#######################
resnet50_stripped = torch.nn.Sequential(*list(resnet50.children())[:-1])
print(resnet50_stripped)

##############testing with random input to get test feature map#############
x = torch.randn([100, 3, 64, 64])
output = resnet50_stripped(x)
print(output.shape)
print(output)

