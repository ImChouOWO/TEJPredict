import pandas as pd 
import matplotlib.pyplot as plt
from collections import Counter


top_list = []
bottom_list = []

top_uniqe = []
bottom_uniqe = []

def load_data(year:str,file_path:str):
    global top_list
    global bottom_list

    data = pd.read_excel(file_path)
    try:
        for i in list(data['top 20%'].values):
            top_list.append(i)

    except:
        for i in list(data['bottom 20%']):
            bottom_list.append(i)
       

def draw_pie(data_list, title):
    # 計算列表中各元素的出現次數
    count = Counter(data_list)

    # 取出現次數最多的前五個元素
    top_five = count.most_common(10)

    # 準備餅圖數據
    labels = [item[0] for item in top_five]
    sizes = [item[1] for item in top_five]

    # 繪製餅圖
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title(title)

    # 顯示圖形
    plt.savefig(f"classifier/outputData/output/img/{title}.png")
      


if __name__ =="__main__":
   
    for i in ["2013","2014","2015","2016","2017","2018","2019","2020","2021","2022"]:
        file_top_path = f"classifier/outputData/output/predictData/final_output_top_{i}.xlsx"
        file_bottom_path = f"classifier/outputData/output/predictData/final_output_bottom_{i}.xlsx"

        load_data(year=i,file_path=file_top_path)
        load_data(year=i,file_path=file_bottom_path)
    draw_pie(top_list, 'Top 10 company')
    draw_pie(bottom_list,"Bottom 10 company")
    
