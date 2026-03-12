# 按照测试集 验证集 还有训练集的txt文件把文件分别复制到对应的文件夹里
import os
import shutil

def transfer_files(txt_file, source_folder, target_folder, label_folder, target_label_folder):
    with open(txt_file, 'r') as f:
        for line in f:
            file_name = line.strip()
            base_name = os.path.basename(file_name)
            source_path = os.path.join(source_folder, base_name)
            taget_path = os.path.join(target_folder, base_name)

            # 找到对应的标签
            label_name = os.path.splitext(base_name)[0] + '.txt' # 根据
            if os.path.exists(source_path):
                shutil.copy(source_path, taget_path)
                # print(f"Copied: {source_path} to {target_path}")
                
            # 复制对应的标签文件
            source_label_path = os.path.join(label_folder, label_name)
            target_label_path = os.path.join(target_label_folder, label_name)
            if os.path.exists(source_label_path):
                shutil.copy(source_label_path, target_label_path)
                # print(f"Copied: {source_label_path} to {target_label_path}")

# 示例用法
source_folder = r'C:\Users\qian gao\git_project\hand_bone_detect\datasets\VOCdevkit\JPEGImages'
target_folder_train = r'C:\Users\qian gao\git_project\hand_bone_detect\datasets\VOCdevkit\images\train'
target_folder_val = r'C:\Users\qian gao\git_project\hand_bone_detect\datasets\VOCdevkit\images\val'
target_folder_test = r'C:\Users\qian gao\git_project\hand_bone_detect\datasets\VOCdevkit\images\test'
txt_file_train = r'C:\Users\qian gao\git_project\hand_bone_detect\datasets\VOCdevkit\train.txt'
txt_file_val = r'C:\Users\qian gao\git_project\hand_bone_detect\datasets\VOCdevkit\val.txt'
txt_file_test = r'C:\Users\qian gao\git_project\hand_bone_detect\datasets\VOCdevkit\test.txt'       
label_folder = r'C:\Users\qian gao\git_project\hand_bone_detect\datasets\VOCdevkit\labels'
target_label_folder_train = r'C:\Users\qian gao\git_project\hand_bone_detect\datasets\VOCdevkit\labels\train'
target_label_folder_val = r'C:\Users\qian gao\git_project\hand_bone_detect\datasets\VOCdevkit\labels\val'
target_label_folder_test = r'C:\Users\qian gao\git_project\hand_bone_detect\datasets\VOCdevkit\labels\test'

# 创建目标文件夹
os.makedirs(target_folder_train, exist_ok=True)
os.makedirs(target_folder_val, exist_ok=True)
os.makedirs(target_folder_test, exist_ok=True)
os.makedirs(target_label_folder_train, exist_ok=True)
os.makedirs(target_label_folder_val, exist_ok=True)
os.makedirs(target_label_folder_test, exist_ok=True)

# 复制文件
transfer_files(txt_file_train, source_folder, target_folder_train, label_folder, target_label_folder_train)
transfer_files(txt_file_val, source_folder, target_folder_val, label_folder, target_label_folder_val)
transfer_files(txt_file_test, source_folder, target_folder_test, label_folder, target_label_folder_test)