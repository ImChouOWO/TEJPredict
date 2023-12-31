import pandas as pd 


def data_preprocess(data_year:str):
    data = pd.read_excel(f"dataset/cleanData/dataYear/data_{data_year}.xlsx")
    data['ROE'] = data['稅前淨利']/(data['資產總額']-data['負債總額'])
    data['ROA'] = data["稅前淨利"]/data['資產總額']
    data=data.drop(columns="年月")
    
    data_combine(data = data, data_year = data_year)
    return data

def data_combine(data:pd.DataFrame,data_year:str):
    predict_data = pd.read_excel(f"predict/predictData/predict_{data_year}.xlsx")
    data["預期 稅後淨利成長率"] = predict_data['預期 稅後淨利成長率']
    data.to_excel(f"classifier/outputData/preprocessData/data_ROE_ROA_predict_{data_year}.xlsx",index=False)

if __name__ =="__main__":
    for i in ["2013","2014","2015","2016","2017","2018","2019","2020","2021","2022"]:
        data = data_preprocess(data_year=i)
   
    
