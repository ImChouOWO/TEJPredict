import pandas as pd 

origin_data_path = "dataset/N電腦周邊2013123120221231.xlsx"

def load_data():
    data = pd.read_excel(origin_data_path)
    return data

def data_clean(data:pd.DataFrame,drop_col:list):

    data = data.drop(columns=drop_col)


    return data

def data_quartile(data: pd.DataFrame, metrics: list) -> pd.DataFrame:
    # 將指定指標的空值以該年份的平均值替代
    for metric in metrics:
        data[metric] = data.groupby(data['年月'].dt.year)[metric].transform(lambda x: x.fillna(x.mean()))

    # 計算四分位數
    for metric in metrics:
        try:
            data[f'{metric}_四分位數'] = data.groupby(data['年月'].dt.year)[metric].transform(
                lambda x: pd.qcut(x, 4, labels=False, duplicates='drop'))
        except ValueError as e:
            print(f"計算 {metric} 的四分位數時發生錯誤: {e}")

    # data.to_excel("dataset/cleanData/dataClean.xlsx")
    return data

def data_splite_by_year(data:pd.DataFrame):
   
    data['年月'] =pd.to_datetime(data['年月'])
    grouped_by_year = [group for _, group in data.groupby(data['年月'].dt.year)]
   
    year_list = ["2013","2014","2015","2016","2017","2018","2019","2020","2021","2022"]
    

    # for i,j in zip(grouped_by_year,year_list):
    #    i.to_excel(f'dataset/cleanData/dataYear/data_{j}.xlsx')

    

    return grouped_by_year


if __name__ == "__main__":
    origin_data = load_data()
    drop_list = ['特別股股本', '預收股款', '待分配股票股利','員工人數']
    data = data_clean(origin_data,drop_list)

    metrics = ['稅前淨利', '稅後淨利成長率', '稅後淨利率', '稅前淨利率', '營業毛利率', '營業利益率']
    data = data_quartile(data,metrics)

    data_year = data_splite_by_year(data)







