
import pandas as pd
geoid_2_od={}
geoid_flow_o={}
geoid_flow_d={}
geoid_id_block4b_2_GISJOIN_1={}
geoid_id_block4b_2_GISJOIN_1_people={}
geoid_id_block4b_2_GISJOIN_1_time={}
df_od_geoid = pd.read_csv(r"C:\Users\Morning\Documents\WeChat Files\wxid_pv2qqr16e4k622\FileStorage\File\2024-03\Boston\cbg_od and shp\OD_Boston city.csv")

df_geoid_oid = pd.read_csv(r"C:\Users\Morning\Documents\WeChat Files\wxid_pv2qqr16e4k622\FileStorage\File\2024-03\spatial join(1).csv")
df_geoid_od = pd.read_csv(r"ver1.csv")

df_people=pd.read_csv(r"C:\Users\Morning\Downloads\building_buffer\OD_Boston city_all.csv")
df_scoial=pd.read_csv(r"C:\Users\Morning\Documents\WeChat Files\wxid_pv2qqr16e4k622\FileStorage\File\2024-03\building_census.csv")
df_x_y=pd.read_csv(r"C:\Users\Morning\Documents\WeChat Files\wxid_pv2qqr16e4k622\FileStorage\File\2024-03\spatial join with location.csv")
sociofactor= ["U7I001", "U7J002", "U7J003", "U7J004", "U7J005", "U7S002", "U7S026", "U8R001", "U8R002", "U8R003", "VBC002", "VBC036"]
x_y_list=['X','Y']

duration_variance= {}
duration_max={}
duration_min={}
social_dict={}
x_y={}
for index, row in df_scoial.iterrows():
    row_dict = {attr: row[attr] for attr in sociofactor}
    social_dict[row['id_block4b']]=row_dict
for index, row in df_x_y.iterrows():
    row_dict = {attr: row[attr] for attr in x_y_list}
    x_y[row['id_block4b']]=row_dict


O_list=[]
D_list=[]
people_list=[]
duration_list=[]
duration_variance_csv_list=[]
duration_max_csv_list=[]
duration_min_csv_list=[]
social_list_dict={}
x_y_dict={}
for index, row in df_geoid_od.iterrows():

    block_id=row['id_block4b']

    social_each_row_dict=social_dict[block_id]
    xy_row_dict=x_y[block_id]
    for label_ in social_each_row_dict:
        if label_ not in social_list_dict:
            social_list_dict[label_]=[]
        social_list_dict[label_].append(social_each_row_dict[label_])

    for label_ in xy_row_dict:
        if label_ not in x_y_dict:
            x_y_dict[label_]=[]
        x_y_dict[label_].append(xy_row_dict[label_])



for label_ in social_list_dict:
    df_geoid_od[label_]=social_list_dict[label_]
for label_ in x_y_dict:
    df_geoid_od[label_]=x_y_dict[label_]

df_geoid_od.to_csv('boston_morphometrics_label_average_std_total.csv')
