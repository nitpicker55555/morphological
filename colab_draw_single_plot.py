from colab_csv import *
import shap
dict_={'label_func': 'Function',
 '2': 'Commercial',
 '3': 'Industrial',
 '4': 'Institutional',
 '5': 'Mixed Use',
 '6': 'Public Open Space',
 '7': 'Residential'}
for i in range(6):
  fig, ax = plt.subplots(figsize=(10, 60))
  shap.summary_plot(shap_values[:,:,i],X,max_display=113)
  # 绘制 SHAP 条形图到指定的 Axes 上
  # shap.plots.bar(shap_values[:, :, i], max_display=113, show=False, ax=ax)

  # 保存图形到文件
  fig.savefig(f"shap_summary_without_weight_{dict_[str(i+2)]}.pdf",bbox_inches='tight', format='pdf')  # 彩图

for i in range(6):
  fig, ax = plt.subplots(figsize=(10, 60))

  # 绘制 SHAP 条形图到指定的 Axes 上
  shap.plots.bar(shap_values[:, :, i], max_display=113, show=False, ax=ax)

  # 保存图形到文件
  fig.savefig(f"shap_bar_plot_without_weight_{dict_[str(i+2)]}.pdf",bbox_inches='tight', format='pdf')  # 柱状图