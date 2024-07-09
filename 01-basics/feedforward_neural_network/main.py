import torch
import torch.nn as nn 
import torchvision 
import torchvision.transforms as transforms

# Device configuration
device1 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 784
hidden_size = 500
num_classes = 50
num_epoch = 5
batch_size = 100
learning_rate =0.001

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='../../data',train=True,transform=transforms.ToTensor(),download=True)
test_dataset = torchvision.datasets.MNIST(root='../../data',train=False,transform=transforms.ToTensor())


# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

# Fully connected neural network with one hidden layer
# num_classes是输出层的神经元数量（即分类数量）
class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(NeuralNet,self).__init__()  # 调用父类 nn.Module 的构造函数，初始化模块
        self.fc1 = nn.Linear(input_size,hidden_size)  # 定义第一层全连接层，将输入映射到隐藏层。
        self.relu = nn.ReLU()  # 定义ReLU激活函数
        self.fc2 = nn.Linear(hidden_size, num_classes) # 定义第二层全连接层，将隐藏层映射到输出层。
    
    # 定义前向传播函数，描述数据如何通过网络
    def forward(self, x): 
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
# 创建 NeuralNet 实例并将其移动到指定的设备（例如CPU或GPU）
model = NeuralNet(input_size, hidden_size, num_classes).to(device1)


# Loss and optimizer
# 交叉熵损失函数（CrossEntropyLoss），常用于分类问题。它计算预测类别概率分布与真实标签之间的差异。
criterion = nn.CrossEntropyLoss()
# Adam优化器（Adaptive Moment Estimation），于梯度下降的优化方法，具有自适应学习率的优点。
# model.parameters(): 模型中所有需要优化的参数。
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader) #训练集中有多少批次（batches）的数据

for epoch in range(num_epoch):
    for i, (images,labels) in enumerate(train_loader):
        # 将图像数据展平为向量（28x28变为784），并移动到指定的设备
        images = images.reshape(-1,28*28).to(device1)
        labels = labels.to(device1)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize 
        optimizer.zero_grad()   # 避免累积梯度
        loss.backward()  # 计算损失函数相对于模型参数的梯度
        optimizer.step()   # 更新模型参数
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epoch, i+1, total_step, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)

# 测试阶段不需要计算梯度,提高速度,节省资源
with torch.no_grad():
    correct = 0
    total = 0
    for images,labels in test_loader:
        images = images.reshape(-1,28*28).to(device1)
        labels = labels.to(device1)
        outputs = model(images)
        
        # 丢掉max 取标签,用来和label对比
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')