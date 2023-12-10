import pandas as pd 

origin_data_path = "dataset/N電腦周邊2013123120221231.xlsx"

def load_data():
    data = pd.read_excel(origin_data_path)
    return data

def data_clean(data:pd.DataFrame,drop_col:list):

    data = data.drop(columns=drop_col)
    if '公司' in data.columns:
        
        data['公司'] = data['公司'].str.split().str[0]

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

    data.to_excel("dataset/cleanData/dataClean.xlsx",index=False)
    return data

def data_splite_by_year(data: pd.DataFrame):
    data['年月'] = pd.to_datetime(data['年月'])

    # 转换日期格式为数值型格式并相加
    data['年月'] = data['年月'].dt.year * 10000 + data['年月'].dt.month * 100 + data['年月'].dt.day

    # 按年份分组
    grouped_by_year = {year: group for year, group in data.groupby(data['年月'] // 10000)}

    year_list = ["2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022"]
    combined_data_2013_2017 = []

    for year in year_list:
        year_data = grouped_by_year.get(int(year), pd.DataFrame())

        # 将 2013 到 2017 年的数据添加到列表中
        if year in ["2013", "2014", "2015", "2016", "2017"]:
            combined_data_2013_2017.append(year_data)

        # 将每年的数据保存为 Excel
        # year_data=year_data.drop(columns="公司")
        year_data.to_excel(f'dataset/cleanData/dataYear/data_{year}.xlsx', index=False)

    # 将 2013 到 2017 年的组合数据合并并保存为一个 Excel
    combined_data_2013_2017_df = pd.concat(combined_data_2013_2017)
    combined_data_2013_2017_df.to_excel('dataset/cleanData/data_2013_to_2017.xlsx', index=False)

    return grouped_by_year




if __name__ == "__main__":
    origin_data = load_data()
    drop_list = ['特別股股本', '預收股款', '待分配股票股利','員工人數']
    data = data_clean(origin_data,drop_list)

    metrics = ['稅前淨利', '稅後淨利成長率', '稅後淨利率', '稅前淨利率', '營業毛利率', '營業利益率']
    data = data_quartile(data,metrics)

    data_year = data_splite_by_year(data)







