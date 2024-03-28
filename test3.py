import csv
import pandas as pd
def extract_values(filename):
    row_dict={}
    sociofactor = ["U7I001", "U7J002", "U7J003", "U7J004", "U7J005", "U7S002", "U7S026", "U8R001", "U8R002", "U8R003",
                   "VBC002", "VBC036"]

    df = pd.read_csv(filename, skipinitialspace=True)

    # 找到“CK”列的索引
    for index, row in df.iterrows():
        row_dict = {attr: row[attr] for attr in sociofactor}
        print(row_dict)
        break
        # if row['GISJOIN_1'] not in geoid_id_block4b_2_GISJOIN_1_people:
    # print(dict_after_ck.keys())

# 替换文件名和列名
filename = r"C:\Users\Morning\Documents\WeChat Files\wxid_pv2qqr16e4k622\FileStorage\File\2024-03\building_census data.csv"
column_name = 'ck'

result_dict = extract_values(filename)
print(result_dict)
