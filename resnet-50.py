import torchvision.models as models
import torch.nn
import torch
import torchvision
import torchvision.transforms as transforms

##############Getting the CIFAR10 dataset######################
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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

