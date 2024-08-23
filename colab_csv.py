# from colab_general import *
import os

import pandas as pd
import shap

import csv

def generate_csv(shap_values,df_100m,X_names,label="0"):
    result_dict = {}
    folder_name="seed"+label
    os.mkdir(folder_name)
    with open(f'plus_time_without_weight.csv', 'r') as csvfile:

        csvreader = csv.reader(csvfile)

        for row in csvreader:
            key = row[5]
            value = row[4]

            result_dict[key] = value

    base_path = f"{folder_name}/shap_values_without_weight_"
    print(shap_values.shape)
    # Generating and saving 6 CSV files
    csv_files = []

    for i in range(6):
        file_path = f"{base_path}{result_dict[str(i + 2)]}{label}.csv"
        data_slice = shap_values[:, :, i].values  # 提取底层的 NumPy 数组
        df = pd.DataFrame(data_slice, columns=X_names)

        df['id_block4b'] = df_100m['id_block4b']
        df['id_block4b'] = df_100m['id_block4b']
        df.to_csv(file_path, index=False)
        csv_files.append(file_path)
# print('finish csv_files')
# from colab_single_pdf import draw_single
# from colab_pdf import draw_summary
# draw_single(shap_values,X)
# print('single')
# draw_summary(shap_values,X)
# print('draw_summary')
