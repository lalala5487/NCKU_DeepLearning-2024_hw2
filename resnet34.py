from tqdm import tqdm
import cv2
import torch
import os
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision import models
from torchvision.models import resnet34
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
import matplotlib.pyplot as plt

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
        im1 = cv2.resize(im1, target_size)
        im1_tensor = torch.from_numpy(im1.transpose(2, 0, 1)).float()  # Convert to float tensor and move to GPU if available
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

model = models.resnet34(pretrained=False,num_classes = 50)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)  

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_accuracies=[]
val_accuracies=[]
train_loss=0.0
best_val_accuracy = 0.0
epochs = 15
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    train_correct = 0
    train_total = 0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=True)  # 创建进度条

    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, torch.argmax(labels, dim=1))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == torch.argmax(labels, dim=1)).sum().item()
    train_loss /= len(val_loader)
    train_accuracy = train_correct / train_total
    # 更新进度条显示训练损失和准确率
    pbar.set_postfix({'Train Loss': epoch_loss / len(train_loader), 'Train Acc': train_correct/train_total})
    train_accuracies.append(train_accuracy)

    # 验证
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, torch.argmax(labels, dim=1))
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == torch.argmax(labels, dim=1)).sum().item()
    
    val_loss /= len(val_loader)
    val_accuracy = val_correct / val_total
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Train acc:{train_accuracy}')
      # 保存最佳模型
    val_accuracies.append(val_accuracy)

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), './hw2result/q2/resnet_best_model.pth')
        print(f"Best model saved with accuracy: {val_accuracy:.4f}")
# 测试评估
model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, torch.argmax(labels, dim=1))
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == torch.argmax(labels, dim=1)).sum().item()

test_loss /= len(test_loader)
test_accuracy = test_correct / test_total
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

# 绘制训练和验证过程中的损失和准确率
plt.plot(range(1, epochs+1), train_accuracies, label='Train accuracy')
plt.plot(range(1, epochs+1), val_accuracies, label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')
plt.savefig("./hw2result/q2/resnet34_result_fig.png")
plt.show()
torch.save(model, './hw2result/q2/resnet34_results.pth')
summary(model)