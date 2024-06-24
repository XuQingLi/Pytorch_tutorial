import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms


# ================================================================== #
#                         Table of Contents                          #
# ================================================================== #

# 1. Basic autograd example 1               (Line 25 to 39)
# 2. Basic autograd example 2               (Line 46 to 83)
# 3. Loading data from numpy                (Line 90 to 97)
# 4. Input pipline                          (Line 104 to 129)
# 5. Input pipline for custom dataset       (Line 136 to 156)
# 6. Pretrained model                       (Line 163 to 176)
# 7. Save and load model                    (Line 183 to 189) 


# ================================================================== #
#                     1. Basic autograd example 1                    #
# ================================================================== #

# Create tensors.
# requires_grad表示这些张量需要计算梯度（即在反向传播时需要计算它们的导数）
x = torch.tensor(1.,requires_grad=True)
w = torch.tensor(2.,requires_grad=True)
b = torch.tensor(3.,requires_grad=True)

# Build a computational graph.
y = w * x + b    # y = 2 * x + 3509

# Compute gradients.
# y.backward() 触发反向传播，计算 y 对各个张量（x, w, b）的梯度，并存储在这些张量的 .grad 属性中。
y.backward()


# Print out the gradients.
print(x.grad)    # x.grad = 2 
print(w.grad)    # w.grad = 1 
print(b.grad)    # b.grad = 1 


# ================================================================== #
#                    2. Basic autograd example 2                     #
# ================================================================== #

# Create tensors of shape (10, 3) and (10, 2).
# x 是形状为 (10, 3) 的张量，表示有 10 个样本，每个样本有 3 个特征。
x = torch.randn(10,3)
y = torch.randn(10,2)

# Build a fully connected layer. 3个输入特征和2个输出特征
# 全连接层，有 3 个输入特征和 2 个输出特征
linear = nn.Linear(3, 2) 
print ('w: ', linear.weight)
print ('b: ', linear.bias)

# Build loss function and optimizer.
# criterion 是均方误差损失函数（MSE Loss），用于衡量预测值和真实值之间的差异。
# optimizer 是随机梯度下降（SGD）优化器，用于更新模型的参数，学习率（lr）设置为 0.01。
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# Forward pass. 输出的是预测的y
pred = linear(x)
pred = linear(x)

# Compute loss.  计算实际y和预测的均方误差     loss.item() 打印损失值

loss = criterion(pred, y)
# loss = nn.MSELoss(pred,y)

print('loss: ', loss.item())

# Backward pass.  计算损失对模型参数（权重和偏置）的梯度。
loss.backward()

# Print out the gradients.   
print ('dL/dw: ', linear.weight.grad) 
print ('dL/db: ', linear.bias.grad)

# 1-step gradient descent.
optimizer.step()

# You can also perform gradient descent at the low level. 
# 展示了如何在不使用优化器的情况下手动执行梯度下降。这实际上是在每个参数上减去学习率乘以梯度。
# linear.weight.data.sub_(0.01 * linear.weight.grad.data)
# linear.bias.data.sub_(0.01 * linear.bias.grad.data)

# Print out the loss after 1-step gradient descent.
pred = linear(x)
loss = criterion(pred, y)
# loss = nn.MSELoss(pred, y)
print('loss after 1 step optimization: ', loss.item())


# ================================================================== #
#                     3. Loading data from numpy                     #
# ================================================================== #

# Create a numpy array.
x = np.array([[1, 2], [3, 4]])

# Convert the numpy array to a torch tensor.   将 NumPy 数组 x 转换为 PyTorch 张量 y
y = torch.from_numpy(x)

# Convert the torch tensor to a numpy array.
z = y.numpy()


# ================================================================== #
#                         4. Input pipeline                           #
# ================================================================== #

# torchvision.datasets.CIFAR10 用于加载 CIFAR-10 数据集。
# root 参数指定数据存储的路径。
# train=True 表示加载训练集。
# transform=transforms.ToTensor() 将图像转换为张量。
# download=True 表示如果数据不存在，则下载数据。
train_dataset = torchvision.datasets.CIFAR10(root='../../data/',train=True,transform=transforms.ToTensor(),download=True)

# Fetch one data pair (read data from disk).  从数据集中获取第一个数据对（图像和标签）
image, label = train_dataset[0]
print (image.size())
print (label)

# Data loader (this provides queues and threads in a very simple way).
# batch_size 指定每个批次的大小，这里是 64。
# shuffle=True 表示在每个 epoch 开始时打乱数据
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)


# When iteration starts, queue and thread start to load data from files.
data_iter = iter(train_loader)
# Mini-batch images and labels.  使用迭代器获取一个 mini-batch 的数据。
images, labels = next(data_iter)


# Actual usage of the data loader is as below.  循环迭代整个数据集的 mini-batch，用于训练模型。
for images,labels in train_loader:
    # Training code should be written here.
    pass

# ================================================================== #
#                5. Input pipeline for custom dataset                 #
# ================================================================== #

# You should build your custom dataset as below.

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
         # TODO
        # 1. Initialize file paths or a list of file names. 
        pass
    def __getitem__():
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        pass
    def  __len__():
        # You should change 0 to the total size of your dataset.
        return 0

# You can then use the prebuilt data loader. 
custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,batch_size=64,shuffle=False)
custom_dataset = CustomDataset()
   
# ================================================================== #
#                        6. Pretrained model                         #
# ================================================================== #

# Download and load the pretrained ResNet-18.  加载预训练模型
resnet = torchvision.models.resnet18(pretrained=True)
# If you want to finetune only the top layer of the model, set as below.
# 冻结 ResNet-18 模型的所有参数，这样在训练过程中这些参数不会被更新
for param in resnet.parameters():
    param.requires_grad = False
# Replace the top layer for finetuning.  替换 ResNet-18 的全连接层（fc），使其输出 100 个类别。
resnet.fc = nn.Linear(resnet.fc.in_features, 100)   # 100 is an example.

# Forward pass.
images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
print(outputs.size())  # (64, 100)


# ================================================================== #
#                      7. Save and load the model                    #
# ================================================================== #

# Save and load the entire model. 保存整个模型到 model.ckpt 文件
torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')


# Save and load only the model parameters (recommended).
# 保存模型的参数到 params.ckpt 文件     加载模型参数
torch.save(resnet.state_dict(), 'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))
