from tqdm import tqdm
import cv2
import torch
import os
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

def load_images(file_path, channel="R"):  
    with open(file_path) as f:
        lines = f.readlines()
    imgs, labels = [], []
    target_size = (128, 128)
    print('Total images', len(lines))
    for i in tqdm(range(len(lines)), desc="Loading images"):
        fn, label = lines[i].strip().split(' ')
        if not os.path.exists(fn):
            print(f"Warning: {fn} does not exist.")
            continue
        im1 = cv2.imread(fn)
        if im1 is None:
            print(f"Warning: {fn} cannot be read.")
            continue
        # print(f"加载的图像 {fn} 的形状: {im1.shape}")
        im1 = cv2.resize(im1, target_size)
        im1 = im1.astype(np.float32) / 255.0  # 将图像转换为浮点数并归一化
        # 拆成各種輸入
        im1_r, im1_g, im1_b = im1[:, :, 0], im1[:, :, 1], im1[:, :, 2]
        if channel == "R":
            im1_r = np.expand_dims(im1_r, axis=2)
            im1_r_tensor = torch.from_numpy(im1_r.transpose(2, 0, 1)).float()  # 转换为浮点数
            imgs.append(im1_r_tensor)
            labels.append(int(label))
        elif channel == "G":
            im1_g = np.expand_dims(im1_g, axis=2)
            im1_g_tensor = torch.from_numpy(im1_g.transpose(2, 0, 1)).float()  # 转换为浮点数
            imgs.append(im1_g_tensor)
            labels.append(int(label))
        elif channel == "B":
            im1_b = np.expand_dims(im1_b, axis=2)
            im1_b_tensor = torch.from_numpy(im1_b.transpose(2, 0, 1)).float()  # 转换为浮点数
            imgs.append(im1_b_tensor)
            labels.append(int(label))
        elif channel == "RG":
            im1_rg = im1[:, :, 0:2]
            im1_rg_tensor = torch.from_numpy(im1_rg.transpose(2, 0, 1)).float()  # 转换为浮点数
            imgs.append(im1_rg_tensor)
            labels.append(int(label))
        elif channel == "GB":
            im1_gb = im1[:, :, 1:3]
            im1_gb_tensor = torch.from_numpy(im1_gb.transpose(2, 0, 1)).float()  # 转换为浮点数
            imgs.append(im1_gb_tensor)
            labels.append(int(label))
        elif channel == "RB":
            im1_rb = im1[:, :, [0, 2]]
            im1_rb_tensor = torch.from_numpy(im1_rb.transpose(2, 0, 1)).float()  # 转换为浮点数
            imgs.append(im1_rb_tensor)
            labels.append(int(label))
        else:
            # 不是上面的就用正常RGB
            im1_tensor = torch.from_numpy(im1.transpose(2, 0, 1)).float()  # 转换为浮点数  # 將通道維度移動到最前面
            imgs.append(im1_tensor)
            labels.append(int(label))

    if not imgs:
        raise ValueError("No valid images found.")
    
    imgs_tensor = torch.stack(imgs)
    labels_tensor = torch.tensor(labels)
    return imgs_tensor, labels_tensor
channel = "B"

x, y = load_images('train.txt',channel)
val_x, val_y = load_images('val.txt',channel)
tx, ty = load_images('test.txt',channel)


y_tensor = torch.nn.functional.one_hot(y, num_classes=50)
val_y_tensor = torch.nn.functional.one_hot(val_y, num_classes=50)
test_y_tensor = torch.nn.functional.one_hot(ty, num_classes=50)
train_dataset = TensorDataset(x, y_tensor)
val_dataset = TensorDataset(val_x, val_y_tensor)
test_dataset = TensorDataset(tx, test_y_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# 定义一个简单的CNN模型
class NaiveCNN(nn.Module):
    def __init__(self):
        super(NaiveCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout(0.2)  # 在第一层卷积后添加 Dropout

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout2 = nn.Dropout(0.2)  # 在第一层卷积后添加 Dropout

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout3 = nn.Dropout(0.2)  # 在第一层卷积后添加 Dropout

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.dropout4 = nn.Dropout(0.2)  # 在第一层卷积后添加 Dropout

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.dropout_fc1 = nn.Dropout(0.3)  # 在第一个全连接层后添加 Dropout
        
        self.fc2 = nn.Linear(1024, 512)
        self.dropout_fc2 = nn.Dropout(0.3)  # 在第二个全连接层后添加 Dropout
        
        self.fc3 = nn.Linear(512, 50)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        # print(f"After conv1: {x.shape}")
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        # print(f"After conv2: {x.shape}")
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        # print(f"After conv3: {x.shape}")
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout4(x)
        # print(f"After conv4: {x.shape}")
        x = self.flatten(x)
        # print(f"After flatten: {x.shape}")
        x = F.relu(self.fc1(x))
        x = self.dropout_fc1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout_fc2(x)
        
        x = self.fc3(x)
        
        return x
    
# 模型训练
model = NaiveCNN()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)  

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()
train_accuracies = []
val_accuracies = []
best_val_accuracy = 0.0

epochs = 15
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=True)

    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, torch.argmax(labels, dim=1))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == torch.argmax(labels, dim=1)).sum().item()
        train_accuracy = correct / total
        pbar.set_postfix({'Train Loss': epoch_loss / len(train_loader), 'Train Acc': correct / total})

    train_accuracies.append(train_accuracy)

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, torch.argmax(labels, dim=1))
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == torch.argmax(labels, dim=1)).sum().item()

    val_accuracy = correct / total
    val_accuracies.append(val_accuracy)
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss/len(train_loader)}, Train Acc: {train_accuracy}, Val Loss: {val_loss/len(val_loader)}, Val Acc: {val_accuracy}')

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), './hw2result/q1/naive/naive_B_best_model.pth')
        print(f"Best model saved with accuracy: {val_accuracy:.4f}")

# 测试集评估
model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, torch.argmax(labels,dim=1))
        test_loss+=loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == torch.argmax(labels, dim=1)).sum().item()

test_loss /= len(test_loader)
test_accuracy = correct / total
print(f'{channel} Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

# Plot training and validation accuracies
plt.figure(figsize=(12, 4))
plt.plot(range(1, epochs+1), train_accuracies, label='Train accuracy')
plt.plot(range(1, epochs+1), val_accuracies, label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')
plt.savefig("./hw2result/q1/naive/naivemodel_B_result_fig.png")
plt.show()
torch.save(model, './hw2result/q1/naive/naivemodel_B_results.pth')
summary(model)