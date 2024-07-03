# 读取矩形框坐标和图片尺寸
import re
def read_rectangles_from_txt(file_path):
    rectangles = []
    with open(file_path, 'r') as file:
        lines = file.readlines()

        for line in lines[:-1]:
            x, y, length, width = map(float, line.split())
            x1 = x - length / 2
            y1 = y - width / 2
            x2 = x + length / 2
            y2 = y + width / 2
            rectangles.append(((x1, y1), (x2, y2)))


        # 解析图片尺寸
        match = re.match(r'\((\d+), (\d+)\)', lines[-1])
        if match:
            image_size = tuple(map(int, reversed(match.groups())))
        else:
            raise ValueError("Invalid format for image size")

    return rectangles, image_size


'''
file_path = "C:/Users/27210/Desktop/yolov5-7.0/runs/detect/exp13/labels/vacant_13.txt"
rectangles, image_size = read_rectangles_from_txt(file_path)
print("矩形框列表:", rectangles)
print("图片尺寸:", image_size)
'''