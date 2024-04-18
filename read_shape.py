import geopandas as gpd
import pandas as pd
from tqdm import tqdm
def read_shapefile(filepath):
    """
    读取Shapefile文件，并返回一个GeoDataFrame。

    参数:
        filepath (str): Shapefile文件的路径。

    返回:
        GeoDataFrame: 包含Shapefile数据的GeoDataFrame对象。
    """
    try:
        # 使用geopandas的read_file函数读取Shapefile
        gdf = gpd.read_file(filepath)
        # print(gdf.head())
        geo_dict=gdf.set_index('NID')['geometry'].to_dict()
        return geo_dict
    except Exception as e:
        print(f"Error reading the shapefile: {e}")


from shapely.geometry import MultiLineString, Point
from shapely.wkt import loads


def find_nearest_geometry(geometry_dict, longitude, latitude):
    # 创建一个点对象
    point = Point(longitude, latitude)

    # 初始化最小距离和最近的几何对象ID
    min_distance = float('inf')
    nearest_id = None

    # 遍历字典中的所有几何对象
    for geom_wkt,jf  in geometry_dict.items():

        distance = point.distance(geom_wkt)

        # 更新最近的几何对象和最小距离
        if distance < min_distance:
            min_distance = distance
            nearest_id = jf
    result_list = [value / min_distance for value in nearest_id]
    return result_list
df_nid = pd.read_csv(r"C:\Users\Morning\Desktop\hiwi\peng\averaged_columns_weekday.csv")
df_id_block = pd.read_csv(r"C:\Users\Morning\Desktop\hiwi\peng\plus_time.csv")


# 将指定的两列转换为字典
id_block4b_dict = df_id_block.set_index('id_block4b')[['X','Y']].apply(tuple, axis=1).to_dict()
# print(id_block4b_dict)

df_nid.set_index(df_nid.columns[-1], inplace=True)

# 转换DataFrame为字典，其中索引作为键，其他列的值组成列表作为值
nid_jf_dict = df_nid.apply(lambda row: row.tolist(), axis=1).to_dict()
# print(nid_jf_dict)
nid_geometry_dict=read_shapefile(r"C:\Users\Morning\Documents\WeChat Files\wxid_pv2qqr16e4k622\FileStorage\File\2024-04\road(1)\road.shp")
geo_jf_dict={}
for nid,jf in nid_jf_dict.items():
    # geo_jf_dict[nid_geometry_dict[nid]]=nid
    geo_jf_dict[nid_geometry_dict[nid]]=jf
block_jf={}
for id_block,xy in tqdm(id_block4b_dict.items()):

        block_jf[id_block]=find_nearest_geometry(geo_jf_dict, *xy)

hour_list=['11','12','13','14','15','16','17','18','19','20','21','22','23','00','01','02','03','04','05','06','07','08','09','10']

for nid, values in block_jf.items():
    # 找到与'nid'匹配的行
    mask = df_id_block['id_block4b'] == nid
    # 为每个元素在相应的行添加新列，列名为索引
    for i, value in enumerate(values):
        df_id_block.loc[mask, f'weekday_{hour_list[i]}'] = value
df_id_block.to_csv('plus_time.csv')
# nearest_geometry_id = find_nearest_geometry(geometry_dict, -71, 42.22)
# print("Nearest geometry ID:", nearest_geometry_id)

# 使用例子：
# gdf = read_shapefile(r"C:\Users\Morning\Documents\WeChat Files\wxid_pv2qqr16e4k622\FileStorage\File\2024-04\road(1)\road.shp")
# print((gdf['geometry']))
