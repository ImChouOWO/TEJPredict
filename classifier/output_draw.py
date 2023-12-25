import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.font_manager import FontProperties

font_path = '/System/Library/Fonts/PingFang.ttc'
chinese_font = FontProperties(fname=font_path)



def create_id_to_name_mapping(nameID_df):
    """
    Create a mapping from company IDs to company names.

    :param nameID_df: DataFrame containing company IDs and names
    :return: Dictionary mapping from company IDs to names
    """
    return dict(zip(nameID_df['ID'], nameID_df['公司名']))

def load_data_and_replace_ids(year, file_path, id_to_name_map):
    """
    Load data from the given file path and replace company IDs with names using the provided mapping.

    :param year: Year of the data
    :param file_path: Path to the data file
    :param id_to_name_map: Dictionary mapping from company IDs to names
    :return: List of company names
    """
    data_list = []
    data = pd.read_excel(file_path)

    try:
        column_name = 'top 20%' if 'top 20%' in data.columns else 'bottom 20%'
        data_list = [id_to_name_map.get(i, i) for i in list(data[column_name].values)]
    except KeyError as e:
        print(f"Error processing file {file_path}: {e}")
    
    return data_list

def draw_pie(data_list, title):
    count = Counter(data_list)
    top_items = count.most_common(10)

    labels = [item[0] for item in top_items]
    sizes = [item[1] for item in top_items]

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title(title, fontproperties=chinese_font)
    plt.savefig(f"classifier/outputData/output/img/{title}.png")
    plt.show()

def main():
    nameID_df = pd.read_excel("dataset/cleanData/nameID.xlsx")
    id_to_name_map = create_id_to_name_mapping(nameID_df)

    top_list = []
    bottom_list = []

    for year in ["2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022"]:
        file_top_path = f"classifier/outputData/output/predictData/final_output_top_{year}.xlsx"
        file_bottom_path = f"classifier/outputData/output/predictData/final_output_bottom_{year}.xlsx"

        top_list.extend(load_data_and_replace_ids(year, file_top_path, id_to_name_map))
        bottom_list.extend(load_data_and_replace_ids(year, file_bottom_path, id_to_name_map))

    draw_pie(top_list, 'Top 10 Companies')
    draw_pie(bottom_list, "Bottom 10 Companies")

if __name__ == "__main__":
    main()

    # draw_pie(top_list, 'Top 10 company')
    # draw_pie(bottom_list,"Bottom 10 company")
    
