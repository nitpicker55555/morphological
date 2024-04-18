
import pandas as pd
from numpy import sort
from pyparsing import nums

geoid_2_od={}
geoid_flow_o={}
geoid_flow_d={}
geoid_id_block4b_2_GISJOIN_1={}
geoid_id_block4b_2_GISJOIN_1_people={}
geoid_id_block4b_2_GISJOIN_1_time={}
df_od_geoid = pd.read_csv(r"C:\Users\Morning\Documents\WeChat Files\wxid_pv2qqr16e4k622\FileStorage\File\2024-03\Boston\cbg_od and shp\OD_Boston city.csv")

df_geoid_oid = pd.read_csv(r"C:\Users\Morning\Documents\WeChat Files\wxid_pv2qqr16e4k622\FileStorage\File\2024-03\spatial join(1).csv")
df_geoid_od = pd.read_csv(r"C:\Users\Morning\Documents\WeChat Files\wxid_pv2qqr16e4k622\FileStorage\File\2024-03\Boston\bostondata_morpho_function\boston_morphometrics_label_average_std.csv")

df_people=pd.read_csv(r"C:\Users\Morning\Downloads\building_buffer\OD_Boston city_all.csv")
df_scoial=pd.read_csv(r'C:\Users\Morning\Documents\WeChat Files\wxid_pv2qqr16e4k622\FileStorage\File\2024-03\building_census data.csv')

sociofactor= ["U7I001", "U7J002", "U7J003", "U7J004", "U7J005", "U7S002", "U7S026", "U8R001", "U8R002", "U8R003", "VBC002", "VBC036"]
def calculate_variance(data):

    if len(data) < 2:
        return 0

    mean = sum(data) / len(data)

    variance_sum = sum((x - mean) ** 2 for x in data)

    variance = variance_sum / (len(data) - 1)

    return variance
duration_dict={}

for index, row in df_people.iterrows():
    if row['GISJOIN_1'] not in geoid_id_block4b_2_GISJOIN_1_people:
        geoid_id_block4b_2_GISJOIN_1_time[row['GISJOIN_1']]=0
        geoid_id_block4b_2_GISJOIN_1_people[row['GISJOIN_1']]=0
        duration_dict[row['GISJOIN_1']]=[]
    geoid_id_block4b_2_GISJOIN_1_people[row['GISJOIN_1']]+=1
    geoid_id_block4b_2_GISJOIN_1_time[row['GISJOIN_1']]+=row['min_duration_seconds']
    duration_dict[row['GISJOIN_1']].append(row['min_duration_seconds'])
duration_variance= {}
duration_max={}
duration_min={}
social_dict={}
for index, row in df_scoial.iterrows():
    row_dict = {attr: row[attr] for attr in sociofactor}
    social_dict[row['GISJOIN_1']]=row_dict
    # print(row_dict)


for i in duration_dict:
    duration_variance[i]=calculate_variance(duration_dict[i])
    duration_max[i]=max(duration_dict[i])
    duration_min[i]=min(duration_dict[i])

    # geoid_2_od[row['GEOID_1']] =row['OID_']
for index, row in df_geoid_oid.iterrows():
    # geoid_2_od[row['GEOID_1']] =row['OID_']
    geoid_2_od[row['id_block4b']] =row['GEOID_1']
    geoid_id_block4b_2_GISJOIN_1[row['id_block4b']]=row['GISJOIN_1']
for index, row in df_od_geoid.iterrows():
    if row['O'] not in geoid_flow_o:
        geoid_flow_o[row['O']]=0
    geoid_flow_o[row['O']] +=row['flow']
    if row['D'] not in geoid_flow_d:
        geoid_flow_d[row['D']] = 0
    geoid_flow_d[row['D']] +=row['flow']
# for index, row in df_od_geoid.iterrows():
#     if row['D'] not in geoid_flow_d:
#         geoid_flow_d[row['D']] = 0
#     geoid_flow_d[row['D']] +=row['flow']
O_list=[]
D_list=[]
people_list=[]
duration_list=[]
duration_variance_csv_list=[]
duration_max_csv_list=[]
duration_min_csv_list=[]
social_list_dict={}
for index, row in df_geoid_od.iterrows():
    O_list.append(geoid_flow_o[geoid_2_od[row['id_block4b']]])
    D_list.append(geoid_flow_d[geoid_2_od[row['id_block4b']]])
    people_list.append(geoid_id_block4b_2_GISJOIN_1_people[geoid_id_block4b_2_GISJOIN_1[row['id_block4b']]])
    duration_list.append(geoid_id_block4b_2_GISJOIN_1_time[geoid_id_block4b_2_GISJOIN_1[row['id_block4b']]])
    duration_variance_csv_list.append(duration_variance[geoid_id_block4b_2_GISJOIN_1[row['id_block4b']]])
    duration_max_csv_list.append(duration_max[geoid_id_block4b_2_GISJOIN_1[row['id_block4b']]])
    duration_min_csv_list.append(duration_min[geoid_id_block4b_2_GISJOIN_1[row['id_block4b']]])
    social_each_row_dict=social_dict[geoid_id_block4b_2_GISJOIN_1[row['id_block4b']]]
    for label_ in social_each_row_dict:
        if label_ not in social_list_dict:
            social_list_dict[label_]=[]
        social_list_dict[label_].append(social_each_row_dict[label_])


    # geoid_flow_d[row['D']] +=row['flow']
df_geoid_od['O_flow']=O_list
df_geoid_od['D_flow']=D_list
df_geoid_od['people']=people_list
df_geoid_od['duration']=duration_list
df_geoid_od['duration_variance']=duration_variance_csv_list
df_geoid_od['duration_max']=duration_max_csv_list
for label_ in social_list_dict:
    df_geoid_od[label_]=social_list_dict[label_]
# df_geoid_od['duration_min']=duration_min_csv_list
df_geoid_od.to_csv('ver1.csv')
aa=nums.sort()

