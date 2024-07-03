import os
import subprocess
import time

def get_txt_file_paths(txt_dir):
    txt_file_paths = []
    for root, dirs, files in os.walk(txt_dir):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                txt_file_paths.append(file_path)
    return txt_file_paths

txt_dir = "C:/Users/27210/Desktop/test-data3/labels"  #预测框数据文件夹
txt_file_paths = get_txt_file_paths(txt_dir)  #获取所有txt文件路径

#print("所有的文本文件路径：", txt_file_paths)

import subprocess
i=1
start_time = time.time()
# 遍历每个文件路径并执行test.py
for txt_file_path in txt_file_paths:
    # 运行命令行命令
    print("第",i,"个文件路径：",txt_file_path)
    subprocess.call("python test.py " + txt_file_path, shell=True)
    i+=1

#计算总用时
end_time = time.time()
print("总用时：", end_time - start_time)

