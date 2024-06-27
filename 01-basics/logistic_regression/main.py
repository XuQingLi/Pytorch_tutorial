import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Hyper-parameters 超参数
input_size = 28 * 28
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset (images and labels)
train_dataset = torchvision.datasets.MNIST(root='../../data',train=True,transform=transforms.ToTensor(),download=True)
test_dataset = torchvision.datasets.MNIST(root='../../data',train=False,transform=transforms.ToTensor())

# Data loader (input pipeline) 通常用于将数据集划分为小批量，以便逐步传递给模型进行训练和测试
# train_loader 是一个迭代器，返回的是一个包含两个元素的元组 (images, labels)：
# images：一个形状为 (batch_size, channels, height, width) 的张量，包含一批次的图像数据。
# labels：一个形状为 (batch_size,) 的张量，包含对应图像的标签。
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)


# Logistic regression model 
# nn.Linear 是一个线性层，也称为全连接层，它实现了线性变换：y = Wx + b。
# 这里的 input_size 是输入特征的维度，num_classes 是输出的类别数。
model = nn.Linear(input_size,num_classes)
model = nn.Linear(input_size, num_classes)

# Loss and optimizer
# nn.CrossEntropyLoss() computes softmax internally

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # 使用随机梯度下降（SGD）作为优化器。 

# Train the model  表示训练数据集被划分为多少个批次
total_step = len(train_loader)

for epoch in range(num_epochs):
    for i,(images,labels) in enumerate(train_loader):   # enumerate枚举
        # Reshape images to (batch_size, input_size) 
        images = images.reshape(-1, input_size)
        
        # Forward pass 
        outputs = model(images)
        # 使用交叉熵损失函数计算预测输出与实际标签之间的损失
        loss = criterion(outputs, labels)  
        
        # Backward and optimize
        optimizer.zero_grad()  # 清零所有参数的梯度
        loss.backward()    # 反向传播，计算梯度
        optimizer.step()   # 使用计算出的梯度更新参数。
 
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model 测试阶段
# In test phase, we don't need to compute gradients (for memory efficiency)
# 在测试阶段，我们不需要计算梯度。这可以节省内存并加快计算速度
with torch.no_grad():
    correct = 0
    total = 0
    for iamges, labels in test_loader:
        images = images.reshape(-1, input_size)  # -1是维度自适应
        outputs = model(images) 
# torch.max 返回每一行的最大值及其索引。 _ 忽略最大值。outputs.data 中包含模型的预测结果，1 表示在列的维度上找最大值。
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
# 布尔张量可以进行求和运算，True 被视为 1，False 被视为 0.
# 累加测试总正确预测数。
        correct += (predicted == labels).sum()
    print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct /total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')

