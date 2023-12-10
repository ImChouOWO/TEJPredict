
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pandas as pd
import os

# 檢查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pd.read_excel("dataset/cleanData/dataCombine.xlsx")

# 選擇相關特徵和目標列
features = data.drop(columns=['稅後淨利成長率'])
target = data['稅後淨利成長率']

# 標準化特徵
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 將數據分割為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# 轉換為 PyTorch 張量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).to(device)

# 創建數據加載器
train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# 定義注意力機制
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, rnn_output):
        # 計算注意力權重
        attention_weights = torch.softmax(self.attention(rnn_output).squeeze(2), dim=1)
        
        # 應用注意力權重（加權求和）
        weighted_output = torch.sum(rnn_output * attention_weights.unsqueeze(-1), dim=1)
        return weighted_output

# RNN 模型
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加時間步的維度

        # 初始化隱藏狀態
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # RNN
        out, _ = self.rnn(x, h0)

        # 注意力機制
        out = self.attention(out)

        # 全連接層
        out = self.fc(out)
        return out

# 繪製訓練損失圖的函數
def save_model(train_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("img\RNN_img\RNN_losses.png")

if __name__ == "__main__":
    # 超參數
    input_size = X_train.shape[1]
    hidden_size = 128
    num_layers = 2
    output_size = 1
    num_epochs = 200000
    learning_rate = 0.001
    rollback_path = 'model/RNN_attention_model/rollback/model_with_attention.pth'
    model_path = 'model\RNN_attention_model\main_model\model_with_attention.pth'

    # 初始化模型
    model = RNN(input_size, hidden_size, num_layers, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 訓練模型
    train_losses = []
    mean_losses = []

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        print("model not exit")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        if epoch % 100 == 0:
            mean_losses.append(sum(train_losses)/len(train_losses))
            train_losses=[]

        if epoch % (num_epochs / 10) == 0:
            torch.save(model.state_dict(), rollback_path)
            print("model has been saved")

    torch.save(model.state_dict(),model_path)
    save_model(mean_losses)
