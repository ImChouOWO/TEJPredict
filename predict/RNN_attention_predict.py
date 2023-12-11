# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# 檢查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pd.read_excel("dataset\cleanData\dataCombine.xlsx" )

# 選擇相關特徵和目標列
features = data.drop(columns=['稅後淨利成長率'])
target = data['稅後淨利成長率']

# 標準化特徵
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 將數據分割為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

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
        # 检查 x 的维度，只在需要时添加时间步的维度
        if x.dim() == 2:
            x = x.unsqueeze(1)  # 添加时间步的维度

        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # RNN
        out, _ = self.rnn(x, h0)

        # 注意力机制
        out = self.attention(out)

        # 全连接层
        out = self.fc(out)
        return out


def predict(model, input_data, scaler):
    # 确保模型处于评估模式
    model.eval()

    # 标准化输入数据
    input_data_scaled = scaler.transform(input_data)

    # 转换为 PyTorch 张量
    input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32).to(device)
    print("input_tensor :",input_tensor.shape)
    # 检查 input_tensor 的维度，并根据需要调整
    if input_tensor.dim() == 2:
        input_tensor = input_tensor.unsqueeze(1)  # 添加时间步的维度

    # 进行预测
    with torch.no_grad():
        output = model(input_tensor)
        return output.cpu().numpy()




def transform_datetime_features(df):
    for col in df.columns:
        if df[col].dtype == 'datetime64[ns]':
            # 計算年、月、日的加總
            total_date_value = df[col].dt.year * 10000 + df[col].dt.month * 100 + df[col].dt.day

            # 將加總後的結果賦值給原始的日期欄位
            df[col] = total_date_value
    print(df.columns)
    return df
def save_predict(data:list,year):
    return_data = []
    for i in data:
        return_data.append(float(i[0]))
    df = pd.DataFrame({"預期 稅後淨利成長率":return_data})
    print(df)
    df.to_excel(f"predict/predictData/predict_{year}.xlsx")


if __name__ == "__main__":
    # 模型超參數
    input_size = X_train.shape[1]
    hidden_size = 128
    num_layers = 2
    output_size = 1

    # 加載訓練好的模型
    model_path = 'model\RNN_attention_model\main_model\model_with_attention.pth'
    model = RNN(input_size, hidden_size, num_layers, output_size).to(device)
    model.load_state_dict(torch.load(model_path))

    # 數據預處理
    # 加載數據並進行標準化
    data = pd.read_excel("dataset\cleanData\dataCombine.xlsx" )
    features = data.drop(columns=['稅後淨利成長率'])
    scaler = StandardScaler()
    scaler.fit(features)
    # print(features.columns)


    for i in ["2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022"]:


        # 加載要進行預測的新數據
        predict_data = pd.read_excel(f"dataset\cleanData\dataYear\data_{i}.xlsx")
        # print(predict_data.columns)

        

        # 轉換 datetime64 數據

        predict_data = transform_datetime_features(predict_data)
        # 從新數據中提取特徵
        predict_features = predict_data.drop(columns=['稅後淨利成長率'], errors='ignore')
        print("predict_features: ",predict_features.shape)
        # 確保特徵順序與訓練時一致
        predict_features = predict_features[features.columns]

        # 進行預測
        predictions = predict(model, predict_features, scaler)
        
        save_predict(predictions,i)