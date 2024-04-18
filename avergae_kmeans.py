import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 读取数据
data = pd.read_csv(r'C:\Users\Morning\Desktop\hiwi\peng\averaged_columns.csv')  # 替换 'your_file.csv' 为你的文件路径

# 转置数据，使每一列变成一个向量
data_transposed = data.transpose()

# 数据标准化
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_transposed)

# 执行 K-means 聚类
kmeans = KMeans(n_clusters=5)  # 假设我们想要分成3个群集
kmeans.fit(data_scaled)

# 获取聚类结果
clusters = kmeans.labels_

# 为每个群集绘制折线图
for i in range(5):  # 假设有3个群集
    plt.figure(figsize=(40, 6))
    cluster_indices = [index for index, cluster_id in enumerate(clusters) if cluster_id == i]
    for index in cluster_indices:
        plt.plot(data.iloc[:, index], label=f'Column {index}')
    plt.title(f'Cluster {i} Line Plots')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(str(i)+'.png')
