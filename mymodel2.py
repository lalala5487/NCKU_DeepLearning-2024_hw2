from tqdm import tqdm
import cv2
import torch
import os
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
import matplotlib.pyplot as plt
import torch.nn.functional as F


def load_images(file_path, target_size=(224, 224)):  
    with open(file_path) as f:
        lines = f.readlines()
    imgs, labels = [], []
    print(f'Total images in {file_path}:', len(lines))
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
        im1_tensor = torch.from_numpy(im1.transpose(2, 0, 1)).float()
        imgs.append(im1_tensor)
        labels.append(int(label))

    if not imgs:
        raise ValueError("No valid images found.")
    
    imgs_tensor = torch.stack(imgs)
    labels_tensor = torch.tensor(labels)
    return imgs_tensor, labels_tensor


x, y = load_images('train.txt')
val_x, val_y = load_images('val.txt')
tx, ty = load_images('test.txt')


y_tensor = torch.nn.functional.one_hot(y, num_classes=50)
val_y_tensor = torch.nn.functional.one_hot(val_y, num_classes=50)
test_y_tensor = torch.nn.functional.one_hot(ty, num_classes=50)
train_dataset = TensorDataset(x, y_tensor)
val_dataset = TensorDataset(val_x, val_y_tensor)
test_dataset = TensorDataset(tx, test_y_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)


class Attention(nn.Module):
    def __init__(self, in_planes, K, init_weight=True):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化到 1x1
        self.net = nn.Conv2d(in_planes, K, kernel_size=1, bias=False)  # 1x1 卷积
        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活函数

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        att = self.avgpool(x)  # 输出形状：批次大小, 通道数, 1, 1
        att = self.net(att).view(x.shape[0], -1)  # 输出形状：批次大小, K
        return self.sigmoid(att)  # 返回经过 Sigmoid 的注意力图
    
class DepthwiseSeparableConvolution(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size=3,stride=1,padding=1):
        super().__init__()
        self.depthwise_conv=nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_ch
        )
        self.pointwise_conv=nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )
        
    def forward(self, x):
        out=self.depthwise_conv(x)
        out=self.pointwise_conv(out)
        return out
    
class AttentionDepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, K, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise_separable_conv = DepthwiseSeparableConvolution(in_ch, out_ch, kernel_size, stride, padding)
        self.attention = Attention(out_ch, K)
        
    def forward(self, x):
        conv_out = self.depthwise_separable_conv(x)
        att = self.attention(conv_out).view(x.shape[0], -1, 1, 1)  # 形状：批次大小, K, 1, 1
        out = conv_out * att  # 元素级相乘
        return out

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = AttentionDepthwiseSeparableConv(in_ch=3, out_ch=32, K=32)
        self.dropout1 = nn.Dropout(0.2)  # 在第一层卷积后添加 Dropout
        self.conv2 = AttentionDepthwiseSeparableConv(in_ch=32, out_ch=64, K=64)
        self.dropout1 = nn.Dropout(0.2)  # 在第一层卷积后添加 Dropout
        self.conv3 = AttentionDepthwiseSeparableConv(in_ch=64, out_ch=128, K=128)
        self.dropout1 = nn.Dropout(0.2)  # 在第一层卷积后添加 Dropout
        self.conv4 = AttentionDepthwiseSeparableConv(in_ch=128, out_ch=256, K=256)
       
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 14 * 14, 50)
        self.dropout_final = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.pool(self.conv4(x))
        
        x = self.flatten(x)
        x = self.dropout_final(x)
        x = self.fc1(x)
        return x

# 模型训练
model = CNN()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)  

optimizer = optim.Adam(model.parameters(), lr=0.001,  weight_decay=1e-5)
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
    
    val_loss /= len(val_loader)
    val_accuracy = correct / total
    val_accuracies.append(val_accuracy)
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
    
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), './hw2result/q2/my_best_model.pth')
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
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

# Plot training and validation accuracies
plt.figure(figsize=(12, 4))
plt.plot(range(1, epochs+1), train_accuracies, label='Train accuracy')
plt.plot(range(1, epochs+1), val_accuracies, label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')
plt.savefig("./hw2result/q2/mymodel2_result_fig.png")
plt.show()
torch.save(model, './hw2result/q2/mymodel2_results.pth')
summary(model)