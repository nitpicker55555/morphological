import os
import shutil


def organize_files(source_folder):
    # 创建目标文件夹
    merged_folder = os.path.join(source_folder, 'merged_files')
    merged_summary_folder = os.path.join(source_folder, 'merged_summary_files')
    merged_other_folder = os.path.join(source_folder, 'merged_other_files')

    non_merged_folder = os.path.join(source_folder, 'non_merged_files')
    non_merged_summary_folder = os.path.join(source_folder, 'non_merged_summary_files')
    non_merged_other_folder = os.path.join(source_folder, 'non_merged_other_files')

    os.makedirs(merged_folder, exist_ok=True)
    os.makedirs(merged_summary_folder, exist_ok=True)
    os.makedirs(merged_other_folder, exist_ok=True)

    os.makedirs(non_merged_folder, exist_ok=True)
    os.makedirs(non_merged_summary_folder, exist_ok=True)
    os.makedirs(non_merged_other_folder, exist_ok=True)

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        file_path = os.path.join(source_folder, filename)

        # 忽略文件夹
        if os.path.isdir(file_path):
            continue

        if filename.endswith('_merged.pdf'):
            if 'summary' in filename:
                shutil.move(file_path, os.path.join(merged_summary_folder, filename))
            else:
                shutil.move(file_path, os.path.join(merged_other_folder, filename))
        else:
            if 'summary' in filename:
                shutil.move(file_path, os.path.join(non_merged_summary_folder, filename))
            else:
                shutil.move(file_path, os.path.join(non_merged_other_folder, filename))


# 调用函数整理文件
source_folder = r'C:\Users\Morning\Downloads\my_files\content\cc'  # 将此路径替换为你的文件夹路径
organize_files(source_folder)
