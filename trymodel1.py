import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.optim as optim
import torchvision.transforms.functional as TF
from torchinfo import summary

import numpy as np
from numpy import linalg as LA
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

def load_RGB_img(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    imgs, labels = [], []
    # imgs = torch.Tensor()
    # labels = []
    target_size=(128, 128)
    print('Total train images:', len(lines))
    for i in tqdm(range(len(lines)), desc="Loading images"):
    # for i in tqdm(range(100), desc="Loading images"):
        fn, label = lines[i].strip().split(' ')
        im1 = cv2.imread(fn)
        # 如果需要進行影像處理，請在這裡添加相應的處理步驟
        im1 = cv2.resize(im1, target_size)

        # 轉換成 PyTorch Tensor
        im1_tensor = torch.from_numpy(im1.transpose(2, 0, 1))  # 將通道維度移動到最前面
        
        imgs.append(im1_tensor) 
        labels.append(int(label))

    imgs_tensor = torch.stack(imgs)  # 將列表中的 Tensor 堆疊成一個整體 Tensor
    labels_tensor = torch.tensor(labels)

    return imgs_tensor, labels_tensor

x_RGB, y_RGB = load_RGB_img('train.txt')
val_x_RGB, val_y_RGB = load_RGB_img('val.txt')
test_x_RGB, test_y_RGB = load_RGB_img('test.txt')


def load_RG_img(file_path,channel = "RG"):
    with open(file_path) as f:
        lines = f.readlines()
    imgs, labels = [], []
    # imgs = torch.Tensor()
    # labels = []
    target_size=(128, 128)
    print('Total train images:', len(lines))
    for i in tqdm(range(len(lines)), desc="Loading images"):
    # for i in tqdm(range(100), desc="Loading images"):
        fn, label = lines[i].strip().split(' ')
        im1 = cv2.imread(fn)
        # 如果需要進行影像處理，請在這裡添加相應的處理步驟
        im1 = cv2.resize(im1, target_size)

        if channel == "RG":
            im1_rg = im1[:, :, 0:2]
            im1_rg_tensor = torch.from_numpy(im1_rg.transpose(2, 0, 1))
            imgs.append(im1_rg_tensor)
            labels.append(int(label))
            # print(im1_rg.shape)
        elif channel == "GB":
            im1_gb = im1[:, :, 1:3]
            im1_gb_tensor = torch.from_numpy(im1_gb.transpose(2, 0, 1))
            imgs.append(im1_gb_tensor)
            labels.append(int(label))
            # print(im1_gb.shape)
        else:
            im1_rb = im1[:, :, [0, 2]]
            im1_rb_tensor = torch.from_numpy(im1_rb.transpose(2, 0, 1))
            imgs.append(im1_rb_tensor)
            labels.append(int(label))
            # print(im1_rb.shape)


    imgs_tensor = torch.stack(imgs)  # 將列表中的 Tensor 堆疊成一個整體 Tensor
    labels_tensor = torch.tensor(labels)

    return imgs_tensor, labels_tensor

x_RG, y_RG = load_RG_img('train.txt',"RG")
val_x_RG, val_y_RG = load_RG_img('val.txt',"RG")
test_x_RG, test_y_RG = load_RG_img('test.txt',"RG")
x_GB, y_GB = load_RG_img('train.txt',"GB")
val_x_GB, val_y_GB = load_RG_img('val.txt',"GB")
test_x_GB, test_y_GB = load_RG_img('test.txt',"GB")
x_RB, y_RB = load_RG_img('train.txt',"RB")
val_x_RB, val_y_RB = load_RG_img('val.txt',"RB")
test_x_RB, test_y_RB = load_RG_img('test.txt',"RB")

def load_R_img(file_path,channel = "R"):
    with open(file_path) as f:
        lines = f.readlines()
    imgs, labels = [], []
    # imgs = torch.Tensor()
    # labels = []
    target_size=(128, 128)
    print('Total train images:', len(lines))
    for i in tqdm(range(len(lines)), desc="Loading images"):
    # for i in tqdm(range(100), desc="Loading images"):
        fn, label = lines[i].strip().split(' ')
        im1 = cv2.imread(fn)
        # 如果需要進行影像處理，請在這裡添加相應的處理步驟
        im1 = cv2.resize(im1, target_size)

        # 把圖片RGB都拆開
        im1_r, im1_g, im1_b = im1[:, :, 0], im1[:, :, 1], im1[:, :, 2]
        if channel == "R":
            im1_r = np.expand_dims(im1_r, axis=2)
            im1_r_tensor = torch.from_numpy(im1_r.transpose(2, 0, 1))
            imgs.append(im1_r_tensor)
            labels.append(int(label))
        elif channel == "G":
            im1_g = np.expand_dims(im1_g, axis=2)
            im1_g_tensor = torch.from_numpy(im1_g.transpose(2, 0, 1))
            imgs.append(im1_g_tensor)
            labels.append(int(label))
        else:
            im1_b = np.expand_dims(im1_b, axis=2)
            im1_b_tensor = torch.from_numpy(im1_b.transpose(2, 0, 1))
            imgs.append(im1_b_tensor)
            labels.append(int(label))
    
    imgs_tensor = torch.stack(imgs)  # 將列表中的 Tensor 堆疊成一個整體 Tensor
    labels_tensor = torch.tensor(labels)

    return imgs_tensor, labels_tensor

x_R, y_R = load_R_img('train.txt',"R")
val_x_R, val_y_R = load_R_img('val.txt',"R")
test_x_R, test_y_R = load_R_img('test.txt',"R")
x_G, y_G = load_R_img('train.txt',"G")
val_x_G, val_y_G = load_R_img('val.txt',"G")
test_x_G, test_y_G = load_R_img('test.txt',"G")
x_B, y_B = load_R_img('train.txt',"B")
val_x_B, val_y_B = load_R_img('val.txt',"B")
test_x_B, test_y_B = load_R_img('test.txt',"B")

class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = K

        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))


    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        # print(x.shape)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)

class Dynamic_conv2d_const_channel(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4,temperature=34, init_weight=True):
        super(Dynamic_conv2d_const_channel, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

        #TODO 初始化
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])


    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):
        softmax_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)
        weight = self.weight.view(self.K, -1)

        aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size*self.out_planes, self.in_planes//self.groups, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output
    

    
class AdaptiveChannelConvModule(nn.Module):
    def __init__(self, out_channels, kernel_size,pool = True):
        super(AdaptiveChannelConvModule, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv = Dynamic_conv2d_const_channel(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.conv_pad = Dynamic_conv2d_const_channel(2, out_channels//2, kernel_size, padding=kernel_size // 2)
        self.bn1_pad = nn.BatchNorm2d(out_channels//2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.pooling = pool
        
    def forward(self, x):
        batch_size, in_channels, height, width = x.shape
        
        if in_channels < self.out_channels:
            
            x_avgpooled = x.mean(dim=1, keepdim=True)        #channel wise avg pooling
            x_maxpooled = x.max(dim=1, keepdim=True).values  #channel wise max pooling
            x_pad = torch.cat([x_avgpooled, x_maxpooled], dim=1)

            if self.pooling:
                x_pad = self.pool(F.relu(self.bn1_pad(self.conv_pad(x_pad))))
                x = self.pool(x)
            else:
                x_pad = F.relu(self.bn1_pad(self.conv_pad(x_pad)))

            # concat input x and featuremaps x_pad
            x = torch.cat([x, x_pad], dim=1)
            
            # padding channel to self.out_channels
            if x.shape[1] < self.out_channels:
                # print("asasdasd")
                x_avgpooled = x.mean(dim=1, keepdim=True)
                x_maxpooled = x.max(dim=1, keepdim=True).values
                x_avgpooled_repeated = x_avgpooled.repeat(1, (self.out_channels - x.shape[1]) // 2, 1, 1)
                x = torch.cat([x, x_avgpooled_repeated], dim=1)
                x_maxpooled = x_maxpooled.repeat(1, self.out_channels - x.shape[1], 1, 1)
                x = torch.cat([x, x_maxpooled], dim=1) 

        # ensure inputs channel size = self.out_channels
        if x.shape[1] > self.out_channels:
                x = x[:, :self.out_channels, :, :]
    
        out = self.conv(x)
        return out



model = AdaptiveChannelConvModule(out_channels=64, kernel_size=3)
input_image = torch.randn(8, 3, 64, 64) 
output = model(input_image)
print(output.shape)  # 應為 (1, 64, 64, 64)

# RG
y_RG_tensor = torch.nn.functional.one_hot(y_RG, num_classes=50)
val_y_RG_tensor = torch.nn.functional.one_hot(val_y_RG, num_classes=50)
test_y_RG_tensor = torch.nn.functional.one_hot(test_y_RG, num_classes=50)

# RB
y_RB_tensor = torch.nn.functional.one_hot(y_RB, num_classes=50)
val_y_RB_tensor = torch.nn.functional.one_hot(val_y_RB, num_classes=50)
test_y_RB_tensor = torch.nn.functional.one_hot(test_y_RB, num_classes=50)

# GB
y_GB_tensor = torch.nn.functional.one_hot(y_GB, num_classes=50)
val_y_GB_tensor = torch.nn.functional.one_hot(val_y_GB, num_classes=50)
test_y_GB_tensor = torch.nn.functional.one_hot(test_y_GB, num_classes=50)

# R
y_R_tensor = torch.nn.functional.one_hot(y_R, num_classes=50)
val_y_R_tensor = torch.nn.functional.one_hot(val_y_R, num_classes=50)
test_y_R_tensor = torch.nn.functional.one_hot(test_y_R, num_classes=50)

# G
y_G_tensor = torch.nn.functional.one_hot(y_G, num_classes=50)
val_y_G_tensor = torch.nn.functional.one_hot(val_y_G, num_classes=50)
test_y_G_tensor = torch.nn.functional.one_hot(test_y_G, num_classes=50)

# B
y_B_tensor = torch.nn.functional.one_hot(y_B, num_classes=50)
val_y_B_tensor = torch.nn.functional.one_hot(val_y_B, num_classes=50)
test_y_B_tensor = torch.nn.functional.one_hot(test_y_B, num_classes=50)

y_RGB_tensor = torch.nn.functional.one_hot(y_RGB, num_classes=50)
val_y_RGB_tensor = torch.nn.functional.one_hot(val_y_RGB, num_classes=50)
test_y_RGB_tensor = torch.nn.functional.one_hot(test_y_RGB, num_classes=50)

# RG
train_dataset_RG = TensorDataset(x_RG, y_RG_tensor)
val_dataset_RG = TensorDataset(val_x_RG, val_y_RG_tensor)
test_dataset_RG = TensorDataset(test_x_RG, test_y_RG_tensor)

# GB
train_dataset_GB = TensorDataset(x_GB, y_GB_tensor)
val_dataset_GB = TensorDataset(val_x_GB, val_y_GB_tensor)
test_dataset_GB = TensorDataset(test_x_GB, test_y_GB_tensor)

# RB
train_dataset_RB = TensorDataset(x_RB, y_RB_tensor)
val_dataset_RB = TensorDataset(val_x_RB, val_y_RB_tensor)
test_dataset_RB = TensorDataset(test_x_RB, test_y_RB_tensor)

# R
train_dataset_R = TensorDataset(x_R, y_R_tensor)
val_dataset_R = TensorDataset(val_x_R, val_y_R_tensor)
test_dataset_R = TensorDataset(test_x_R, test_y_R_tensor)

# G 
train_dataset_G = TensorDataset(x_G, y_G_tensor)
val_dataset_G = TensorDataset(val_x_G, val_y_G_tensor)
test_dataset_G = TensorDataset(test_x_G, test_y_G_tensor)

# B
train_dataset_B = TensorDataset(x_B, y_B_tensor)
val_dataset_B = TensorDataset(val_x_B, val_y_B_tensor)
test_dataset_B = TensorDataset(test_x_B, test_y_B_tensor)

# RGB
train_dataset_RGB = TensorDataset(x_RGB, y_RGB_tensor)
val_dataset_RGB = TensorDataset(val_x_RGB, val_y_RGB_tensor)
test_dataset_RGB = TensorDataset(test_x_RGB, test_y_RGB_tensor)

batch_size = 128 
shuffle = True  

# RG
train_RG_loader = DataLoader(train_dataset_RG, batch_size=batch_size, shuffle=shuffle)
val_RG_loader = DataLoader(val_dataset_RG, batch_size=batch_size, shuffle=shuffle)
test_RG_loader = DataLoader(test_dataset_RG, batch_size=batch_size, shuffle=shuffle)

# RB
train_RB_loader = DataLoader(train_dataset_RB, batch_size=batch_size, shuffle=shuffle)
val_RB_loader = DataLoader(val_dataset_RB, batch_size=batch_size, shuffle=shuffle)
test_RB_loader = DataLoader(test_dataset_RB, batch_size=batch_size, shuffle=shuffle)

# GB
train_GB_loader = DataLoader(train_dataset_GB, batch_size=batch_size, shuffle=shuffle)
val_GB_loader = DataLoader(val_dataset_GB, batch_size=batch_size, shuffle=shuffle)
test_GB_loader = DataLoader(test_dataset_GB, batch_size=batch_size, shuffle=shuffle)

# R
train_R_loader = DataLoader(train_dataset_R, batch_size=batch_size, shuffle=shuffle)
val_R_loader = DataLoader(val_dataset_R, batch_size=batch_size, shuffle=shuffle)
test_R_loader = DataLoader(test_dataset_R, batch_size=batch_size, shuffle=shuffle)

# G 
train_G_loader = DataLoader(train_dataset_G, batch_size=batch_size, shuffle=shuffle)
val_G_loader = DataLoader(val_dataset_G, batch_size=batch_size, shuffle=shuffle)
test_G_loader = DataLoader(test_dataset_G, batch_size=batch_size, shuffle=shuffle)

# B 
train_B_loader = DataLoader(train_dataset_B, batch_size=batch_size, shuffle=shuffle)
val_B_loader = DataLoader(val_dataset_B, batch_size=batch_size, shuffle=shuffle)
test_B_loader = DataLoader(test_dataset_B, batch_size=batch_size, shuffle=shuffle)

# RGB
train_RGB_loader = DataLoader(train_dataset_RGB, batch_size=batch_size, shuffle=shuffle)
val_RGB_loader = DataLoader(val_dataset_RGB, batch_size=batch_size, shuffle=shuffle)
test_RGB_loader = DataLoader(test_dataset_RGB, batch_size=batch_size, shuffle=shuffle)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # self.conv1 = Dynamic_conv2d(out_planes=32, kernel_size=3, stride=1, padding=1)
        # self.conv2 = Dynamic_conv2d(out_planes=64, kernel_size=3, stride=1, padding=1)
        # self.conv3 = Dynamic_conv2d(out_planes=128, kernel_size=3, stride=1, padding=1)
        # self.conv4 = Dynamic_conv2d(out_planes=256, kernel_size=3, stride=1, padding=1)
        # self.conv5 = Dynamic_conv2d(out_planes=512, kernel_size=3, stride=1, padding=1)

        self.conv1 = AdaptiveChannelConvModule(out_channels=32, kernel_size=3,pool = False)
        self.conv2 = AdaptiveChannelConvModule(out_channels=64, kernel_size=3,pool = False)
        self.conv3 = AdaptiveChannelConvModule(out_channels=128, kernel_size=3,pool = False)
        self.conv4 = AdaptiveChannelConvModule(out_channels=256, kernel_size=3,pool = False)
        self.conv5 = AdaptiveChannelConvModule(out_channels=512, kernel_size=3)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(512 * 2 * 2 , 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 50) 

        self.dropout = nn.Dropout(0.5)  # onlt RGB = 0.3 // all channel = 0.5

    def forward(self, x):
        # print(x.shape)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # print(x.shape)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        # print(x.shape)
        # x = x.view(x.size(0), -1)
        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
model = CNNModel()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 檢查是否有 GPU，有則使用 GPU
model.to(device)  # 將模型移動到設備上

summary(model)

def train(model,traindataloader,valdataLoader,criterion,optimizer,traing_acc,val_acc,epoch,channel):
    model.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0

    max_val_acc = 0
    
    running_loss = 0.0
    for inputs, labels in tqdm(traindataloader):
        # print(inputs[0].shape)
        inputs, labels = inputs.float().to(device), labels.float().to(device)  # 將數據移動到設備上

        # 正向傳播
        outputs = model(inputs)
        # print(type(outputs))
        loss = criterion(outputs, labels)

        # 反向傳播和優化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        labels = labels.argmax(dim=1)
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        running_loss += loss.item() * inputs.size(0)
    
    traing_acc.append(100 * train_correct / train_total)

    model.eval()  # 將模型設置為評估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in valdataLoader:
            inputs, labels = inputs.float().to(device), labels.float().to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            labels = labels.argmax(dim=1)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    val_acc.append(100 * correct / total)

    if (correct / total) > max_val_acc and channel == "RGB":
        torch.save(model.state_dict(), './hw2result/q1/my/mymodel.pth')
        max_val_acc = (correct / total)


    epoch_loss = running_loss / len(traindataloader.dataset)
    print(f"Epoch {epoch + 1}/{10}, Loss: {epoch_loss:.4f}, channel = {channel}, Training Accuracy: {100 * train_correct / train_total:.2f}%, Validation Accuracy: {100 * correct / total:.2f}%")

def plot_acc(train_acc,val_acc,channel):
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"./hw2result/q1/my/myModel_{channel}_accuracy_plot.png")
    plt.show()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# for param in model.parameters():
#     param = param.to(device)

# 訓練循環
epochs = 10  # 設定訓練輪數

train_accuracy_RGB = []
val_accuracy_RGB = []

train_accuracy_RG = []
val_accuracy_RG = []

train_accuracy_RB = []
val_accuracy_RB = []

train_accuracy_GB = []
val_accuracy_GB = []

train_accuracy_R = []
val_accuracy_R = []

train_accuracy_G = []
val_accuracy_G = []

train_accuracy_B = []
val_accuracy_B = []

for epoch in range(epochs):
    model.train()  # 將模型設置為訓練模式

    train(model,train_R_loader,val_R_loader,criterion,optimizer,train_accuracy_R,val_accuracy_R,epoch,"R")
    train(model,train_RG_loader,val_RG_loader,criterion,optimizer,train_accuracy_RG,val_accuracy_RG,epoch,"RG")
    train(model,train_G_loader,val_G_loader,criterion,optimizer,train_accuracy_G,val_accuracy_G,epoch,"G")
    train(model,train_GB_loader,val_GB_loader,criterion,optimizer,train_accuracy_GB,val_accuracy_GB,epoch,"GB")
    train(model,train_B_loader,val_B_loader,criterion,optimizer,train_accuracy_B,val_accuracy_B,epoch,"B")
    train(model,train_RB_loader,val_RB_loader,criterion,optimizer,train_accuracy_RB,val_accuracy_RB,epoch,"RB")
    train(model,train_RGB_loader,val_RGB_loader,criterion,optimizer,train_accuracy_RGB,val_accuracy_RGB,epoch,"RGB")

    
    
plot_acc(train_accuracy_RGB,val_accuracy_RGB,"RGB")
plot_acc(train_accuracy_RG,val_accuracy_RG,"RG")
plot_acc(train_accuracy_GB,val_accuracy_GB,"GB")
plot_acc(train_accuracy_RB,val_accuracy_RB,"RB")
plot_acc(train_accuracy_R,val_accuracy_R,"R")
plot_acc(train_accuracy_G,val_accuracy_G,"G")
plot_acc(train_accuracy_B,val_accuracy_B,"B")

def test(model,dataloader,channel = "RGB"):
    correct = 0
    total = 0
    for inputs, labels in tqdm(dataloader, desc="Evaluating"):
        inputs, labels = inputs.float().to(device), labels.float().to(device)
        labels = labels.argmax(dim=1)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"{channel}-channel Test Accuracy: {accuracy:.4f}")

model_test_path = f"./hw2result/q1/my/mymodel.pth"
model_test = CNNModel()
model_test.load_state_dict(torch.load(model_test_path))
model_test.to(device)
model_test.eval()
#test_RGB
test(model_test,test_RGB_loader,"RGB")

#test RG
test(model_test,test_RG_loader,"RG")

#test RB
test(model_test,test_RB_loader,"RB")

#test GB
test(model_test,test_GB_loader,"GB")

#test R
test(model_test,test_R_loader,"R")

#test G
test(model_test,test_G_loader,"G")

#test R
test(model_test,test_B_loader,"B")
summary(model)