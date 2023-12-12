import pandas as pd
from sklearn.cluster import KMeans

# 讀取資料檔案
file_path = 'classifier/outputData/preprocessData/data_ROE_ROA_predict_2013.xlsx'
data = pd.read_excel(file_path)

# 選取 ROE, ROA, 預期淨利潤成長率
features = data[['ROE', 'ROA', '預期 稅後淨利成長率','稅後淨利成長率']]

# 進行 K-means 分群，分為兩群：前20%與後20%
kmeans = KMeans(n_clusters=2)
kmeans.fit(features)
clusters = kmeans.predict(features)

# 將分群結果新增回原始 DataFrame
data['Cluster'] = clusters

# 獲取前20%和後20%的公司
sorted_data = data.sort_values(by='Cluster')
top_20_percent = sorted_data.head(len(data) // 5)
bottom_20_percent = sorted_data.tail(len(data) // 5)

# 提取公司編碼
top_20_percent_companies = top_20_percent['公司']
bottom_20_percent_companies = bottom_20_percent['公司']

# 輸出結果
print("前 20% 的公司編碼:\n", top_20_percent_companies)
print("\n後 20% 的公司編碼:\n", bottom_20_percent_companies)
