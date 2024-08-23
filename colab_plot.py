from colab_general import *
import shap
shap_grouped_label=['Morphology','OD','Human mobility','Social economy','Latitude & longitude','Road speed holiday','Road speed weekday']
def group_data(shap_values,mode='shap_values'):
  if mode!='x':
    shap_grouped = np.zeros((4695,7,6))
    shap_grouped[:,0,:] = np.sum(shap_values[:,0:45,:],axis=1)
    shap_grouped[:,1,:] = np.sum(shap_values[:,45:47,:],axis=1)
    shap_grouped[:,2,:] = np.sum(shap_values[:,47:51,:],axis=1)
    shap_grouped[:,3,:] = np.sum(shap_values[:,51:63,:],axis=1)
    shap_grouped[:,4,:] = np.sum(shap_values[:,63:65,:],axis=1)
    shap_grouped[:,5,:] = np.sum(shap_values[:,65:89,:],axis=1)
    shap_grouped[:,6,:] = np.sum(shap_values[:,89:113,:],axis=1)
  else:
    print(mode)
    shap_grouped = pd.DataFrame(index=shap_values.index, columns=range(7)).astype(float)

    # 分片并计算总和，保持数据类型不变
    shap_grouped.iloc[:, 0] = shap_values.iloc[:, 0:45].sum(axis=1).astype(float)

    shap_grouped.iloc[:, 1] = shap_values.iloc[:, 45:47].sum(axis=1).astype(float)
    shap_grouped.iloc[:, 2] = shap_values.iloc[:, 47:51].sum(axis=1).astype(float)
    shap_grouped.iloc[:, 3] = shap_values.iloc[:, 51:63].sum(axis=1).astype(float)
    shap_grouped.iloc[:, 4] = shap_values.iloc[:, 63:65].sum(axis=1).astype(float)
    shap_grouped.iloc[:, 5] = shap_values.iloc[:, 65:89].sum(axis=1).astype(float)
    shap_grouped.iloc[:, 6] = shap_values.iloc[:, 89:113].sum(axis=1).astype(float)
  return shap_grouped

import copy
shap_values_copy=copy.deepcopy(shap_values)
shap_values.feature_names=shap_grouped_label
shap_values_grouped=group_data(shap_values_copy.values)


shap_values.values=shap_values_grouped
X_grouped=group_data(X,'x')
shap_values.data=X_grouped


X_grouped=group_data(X,'x')
dict_={'label_func': 'Function',
 '2': 'Commercial',
 '3': 'Industrial',
 '4': 'Institutional',
 '5': 'Mixed Use',
 '6': 'Public Open Space',
 '7': 'Residential'}
X_grouped = pd.DataFrame(X_grouped)

X_grouped.columns = shap_grouped_label
for i in range(6):
  fig, ax = plt.subplots(figsize=(10, 60))
  shap.summary_plot(shap_values_grouped[:,:,i],X_grouped,max_display=113)

  fig.savefig(f"shap_summary_with_weight_{dict_[str(i+2)]}.pdf",bbox_inches='tight', format='pdf')  # 彩色图



for i in range(6):
  fig, ax = plt.subplots(figsize=(8, 8))

  # 绘制 SHAP 条形图到指定的 Axes 上
  shap.plots.bar(shap_values[:, :, i], max_display=113, show=False, ax=ax)

  feature_labels = [tick.get_text() for tick in ax.get_yticklabels()]
  print(f"图 {i+1} 的特征标签:")
  print(feature_labels)
  # 保存图形到文件
  fig.savefig(f"shap_bar_plot_with_weight_{dict_[str(i+2)]}.pdf",bbox_inches='tight', format='pdf')  # 红色柱状图