import pandas as pd
import numpy as np


def compare_csv(file1, file2):
    # 读取两个CSV文件
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # 比较DataFrame的形状是否相同
    if df1.shape != df2.shape:
        print("The files have different shapes.")
        return False

    # 比较DataFrame中的数值是否相同
    comparison = np.isclose(df1.values, df2.values, equal_nan=True)

    if comparison.all():
        print("The files are identical.")
        return True
    else:
        print("The files are different.")
        return False


# 使用例子
file1 = r"C:\Users\Morning\Downloads\shap_values_without_weight_Commercial2.csv"
file2 = r"C:\Users\Morning\Downloads\shap_values_without_weight_Commercial (15).csv"
compare_csv(file1, file2)
