import pandas as pd
from datetime import datetime, timedelta



def is_weekday(date_str):
    """
    判断给定的ISO格式日期时间字符串是否为工作日。

    参数:
    date_str (str): 日期时间字符串，格式为 'YYYY-MM-DDTHH:MM:SSZ'

    返回:
    bool: 如果是周一到周五，则返回True，否则返回False
    """
    # 移除末尾的'Z'并转换为datetime对象

    date_time = datetime.fromisoformat(date_str[:-1])

    # datetime.weekday()返回的是星期几，0是周一，6是周日
    # 如果weekday()返回的是0到4，那么它是工作日
    return 0 <= date_time.weekday() <= 4
# 加载CSV文件
df = pd.read_csv(r"C:\Users\Morning\Downloads\202404_Boston\202404_Boston\here_data_speed_Boston_jf.csv")
def change_time(date_str):

    dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")

    # 减去四小时
    new_dt = dt - timedelta(hours=4)
    return new_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
def judge_hour(time_str):
    col_datetime =datetime.fromisoformat(time_str[:-1])
    return  col_datetime.strftime('%H')
column_names = []  # 用于存储新的列名

current_hour_weekday={}
current_hour_holiday={}

for start_col in range(22, len(df.columns)):  # 从第二列开始，步长为5
    # 选择当前五列

    #
    # selected_columns = df.iloc[:, start_col:start_col+60]
    # # 计算这五列的平均值，并添加到列表中

    try:

        col_name=change_time(df.columns[start_col])
    except:
        col_name=change_time(df.columns[start_col].replace('.1',''))


    if judge_hour(col_name) not in current_hour_holiday and not is_weekday(col_name):
        current_hour_holiday[judge_hour(col_name) ]=[]
    if not is_weekday(col_name):
        current_hour_holiday[judge_hour(col_name)].append(start_col)
    if judge_hour(col_name) not in current_hour_weekday and  is_weekday(col_name):
        current_hour_weekday[judge_hour(col_name) ]=[]
    if is_weekday(col_name):
        current_hour_weekday[judge_hour(col_name)].append(start_col)



averages = []  # 用于存储计算的平均值
for hour_str in current_hour_weekday:


        averages.append(df.iloc[:,current_hour_weekday[hour_str]].mean(axis=1))

        column_names.append(str(hour_str))



print(len(column_names))

averages_df = pd.concat(averages, axis=1)
averages_df.columns = column_names  # 设置新的列名
averages_df['NID']=df['NID']

averages_df.to_csv('averaged_columns_weekday.csv', index=False)

