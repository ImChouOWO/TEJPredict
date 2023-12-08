import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import os

class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()
        self.feature_dim = feature_dim
        self.attention_weights = nn.Parameter(torch.Tensor(feature_dim))

        # Initialize weights
        nn.init.uniform_(self.attention_weights.data, -0.1, 0.1)

    def forward(self, x):
        # Apply softmax to attention weights
        weights = F.softmax(self.attention_weights, dim=0)

        # Apply attention weights
        x = x * weights

        return x

class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.attention = Attention(50)
        self.fc2 = nn.Linear(50, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.attention(x)
        x = self.fc2(x)
        return x







def data_preprocess():
    data_list = []
    load_path_year = ["2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]

    for i in load_path_year:
        df = pd.read_excel(f"dataset/cleanData/dataYear/data_{i}.xlsx")
        data_list.append(df)

    data = pd.concat(data_list, ignore_index=True)

    if '年月' in data.columns:
        base_date = pd.Timestamp('2000-01-01')
        data['年月'] = (pd.to_datetime(data['年月']) - base_date).dt.days
    # data.to_excel("dataset/cleanData/dataCombine.xlsx")
    return data


def draw_mse_chart(mse, num_epochs):
    plt.figure(figsize=(10, 6))
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, mse, label='MSE')
    # plt.scatter(epochs, mse)  # 標記每個點
    # for i, txt in enumerate(mse):
    #     plt.annotate(f'{txt:.2f}', (epochs[i], mse[i]))  # 添加數據點座標標籤
    plt.title('Mean Squared Error Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)  # 添加格線
    plt.legend()
    plt.savefig("img/mse.png")
    

def draw_mae_chart(mae, num_epochs):
    plt.figure(figsize=(10, 6))
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, mae, label='MAE')
    # plt.scatter(epochs, mae)  # 標記每個點
    # for i, txt in enumerate(mae):
    #     plt.annotate(f'{txt:.2f}', (epochs[i], mae[i]))  # 添加數據點座標標籤
    plt.title('Mean Absolute Error Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.grid(True)  # 添加格線
    plt.legend()
    plt.savefig("img/mae.png")
    

def draw_chart_losses(epoch_losses, num_epochs):
    plt.figure(figsize=(10, 6))
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, epoch_losses, label='Epoch Loss')
    # plt.scatter(epochs, epoch_losses)  # 標記每個點
    # for i, txt in enumerate(epoch_losses):
    #     plt.annotate(f'{txt:.2f}', (epochs[i], epoch_losses[i]))  # 添加數據點座標標籤
    plt.title('Training Loss Per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)  # 添加格線
    plt.legend()
    plt.savefig("img/losses.png")
    






if __name__ == "__main__":
    data = data_preprocess()


    epoch_losses = []
    epoch_mse = []
    epoch_mae = []

    epoch_avg_losses = []
    epoch_avg_mse = []
    epoch_avg_mae = []
    
    save_path = "model/model.pth"
    rollback_save_path = "model/rollback/"
    



    # 定義特徵列和目標列
    target_columns = ['稅後淨利成長率', '稅後淨利率', '稅前淨利率', '營業毛利率', '營業利益率']
    feature_columns = [col for col in data.columns if col not in target_columns]

    X = data[feature_columns].values          # 選擇特徵列
    y = data[target_columns].values   # 選擇目標列
    
    # 分割數據
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 標準化數據
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 轉換為 PyTorch 張量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # 創建數據加載器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 模型、損失函數和優化器
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)


    model = Net(input_size=len(feature_columns), output_size=len(target_columns)).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 1000
    for epoch in range(num_epochs):
        if os.path.exists(save_path):
            model.load_state_dict(torch.load(save_path))
            print("Model loaded")
        else:
            print("Model does not exist")
        total_loss = 0.0
        mse_loss, mae_loss = 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向傳播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()


            mse_loss += F.mse_loss(outputs, targets, reduction='sum').item()
            mae_loss += F.l1_loss(outputs, targets, reduction='sum').item()





            # 反向傳播和優化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss = total_loss / len(train_loader)
        avg_mse = mse_loss / len(train_loader.dataset)
        avg_mae = mae_loss / len(train_loader.dataset)

        epoch_losses.append(avg_loss)
        epoch_mse.append(avg_mse)
        epoch_mae.append(avg_mae)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        if epoch % num_epochs//10  == 0 or epoch == num_epochs - 1:

            epoch_avg_losses.append(sum(epoch_losses)/len(epoch_losses))
            epoch_avg_mse.append(sum(epoch_mse)/len(epoch_mse))
            epoch_avg_mae.append(sum(epoch_mae)/len(epoch_mae))

            epoch_losses = []
            epoch_mse = []
            epoch_mae = []



            torch.save(model.state_dict(), f"{rollback_save_path}{epoch}_model.pth")
    
            # 儲存模型
            torch.save(model.state_dict(), save_path)
            print("Model saved as model.pth")





    # 效能評估
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor.to(device))
        test_loss = criterion(y_pred, y_test_tensor.to(device))
    print(f'Test Loss: {test_loss.item():.4f}')
    draw_chart_losses(epoch_avg_losses, len(epoch_avg_losses))
    draw_mse_chart(epoch_avg_mse, len(epoch_avg_mse))
    draw_mae_chart(epoch_avg_mae, len(epoch_avg_mae))



 





    
