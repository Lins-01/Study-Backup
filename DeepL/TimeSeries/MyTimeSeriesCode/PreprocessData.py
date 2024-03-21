import os

def rename_files(root_folder):
    # 遍历root_folder下的所有子目录
    for subdir in next(os.walk(root_folder))[1]:
        subdir_path = os.path.join(root_folder, subdir)
        print("root_folder: ", root_folder)
        print("subdir: ", subdir)
        print("subdir_path: ", subdir_path)
        # 遍历子目录下的所有文件
        for file in os.listdir(subdir_path):
            # 检查文件扩展名是否为.csv
            if file.endswith('.csv'):
                # 构造新的文件名
                new_file_name = f"{file.split('.')[0]}_{subdir}.csv"
                print("file: ", file)
                print("new_file_name: ", new_file_name)
                # 重命名文件
                # 第一个参数是需要重命名的文件的路径/文件名，第二个参数是新的文件名/路径
                os.rename(os.path.join(subdir_path, file), os.path.join(subdir_path, new_file_name))
            # os._exit(0)


# 主函数部分
if __name__ == "__main__":
    # 将此路径替换为您的Operation_csv_data目录的实际路径
    root_folder = ".\\1"
    rename_files(root_folder)