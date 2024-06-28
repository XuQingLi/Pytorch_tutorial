import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# Hyper-parameters 超参数
input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.001 

input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.001

# Toy dataset
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
                    [9.779], [6.182], [7.59], [2.167], [7.042], 
                    [10.791], [5.313], [7.997], [3.1]],dtype=np.float32)
y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
                    [3.366], [2.596], [2.53], [1.221], [2.827], 
                    [3.465], [1.65], [2.904], [1.3]],dtype=np.float32)

# Linear regression model 线性回归模型
model = nn.Linear(input_size, output_size)

# Loss and optimizer   定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model  
for epoch in range(num_epochs):
    # Convert numpy arrays to torch tensors  将array转换为张量
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)

    # Forward pass  前向传播 即输入数据通过模型得到预测结果。 计算Loss
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward and optimize  
    # 优化器清除之前的梯度，避免累积梯度 
    optimizer.zero_grad()
    loss.backward()   # 反向传播，计算损失函数关于模型参数的梯度。
    optimizer.step()  # 根据计算得到的梯度，更新模型的参数。
    
    if(epoch+1) % 5 ==0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))  # loss.item()是获取损失张量的数值

# matplotlib库绘制图形
# from_numpy将训练数据转换为PyTorch张量，然后将其输入模型中进行预测。
# .detach()将结果从当前的计算图中分离出来，这样在将结果转换为NumPy数组时就不会跟踪梯度。
# .numpy()方法将PyTorch张量转换为NumPy数组，以便于后续的绘图操作。
predicted = model(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train, y_train, 'ro', label="Original data")
plt.plot(x_train, predicted, label='Fitter line')
# 调用matplotlib的legend函数显示图例，图例会根据之前plot函数中设置的label来显示。
plt.legend()
plt.show()

# Save the model checkpoint  model.state_dict()方法会返回一个包含模型所有参数的字典
torch.save(model.state_dict(), 'model.ckpt')
