
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import math
import get_rectangle_data # type: ignore
import os
import sys
import time
import threading


all_img_flag=True #False表示只对单张图片进行行列拟合，True表示对整个文件夹下所有图片进行行列拟合

####################################读取整个文件夹下所有图片进行行列拟合#################################################################
if all_img_flag:
    #读取命令行参数
    if len(sys.argv)!= 2:
        print("用法: python test_6_3.py <image_path>")
        sys.exit(1)
    # 读取图像路径
    img_path = sys.argv[1]
    # 定义一个函数，将终端打印的信息追加到指定的文件中
    def redirect_output_to_files(normal_file, error_file):
        sys.stdout = open(normal_file, 'a')  # 以追加模式打开普通输出文件
        sys.stderr = open(error_file, 'a')   # 以追加模式打开错误输出文件
    # 调用函数，将输出追加到文件中
    redirect_output_to_files("C:/Users/27210/Desktop/test_8/normal_output.txt", "c:/Users/27210/Desktop/test_8/error_output.txt")
    ######################################################################################################################################


####################################读取单张图片进行行列拟合##############################################  
else:
    # 读取图像路径
    img_path="C:/Users/27210/Desktop/test-data3/labels/vacant_1370.txt"

#读取矩形框坐标对列表和图像大小
img_name= os.path.splitext(os.path.basename(img_path))[0]  # 获取图像名
coordinates,img_size=get_rectangle_data.read_rectangles_from_txt(img_path) # 存储矩形框坐标对列表和图像大小
save_dir = "C:/Users/27210/Desktop/test_8" # 指定保存目录
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
print(img_name)
print("img_size:",img_size)

#设定行数(暂时只针对6*5的行列进行拟合)
nums_of_row=6
#设定列数(暂时只针对6*5的行列进行拟合)
nums_of_col=5

# 创建绘图对象和子图，设置图像大小
fig, ax = plt.subplots(figsize=(img_size[0]/100, img_size[1]/100)) 
#确定最大相交面积占比阈值
max_intersection_area_ratio=0.01467314308844363
#判定每列前一二个点是否拟合正确的斜率相对误差阈值（百分比）
col_slope_error_threshold= 69.0661

#创建存储中心坐标的空列表
center_xy=[]
# 创建一个字典来存储中心坐标与矩形坐标对的对应关系
center_to_rectangle = {}
#创建一个字典来存储矩形坐标对与中心坐标的对应关系
rectangle_to_center = {}
#创建一个列表来存储列直线的斜率
col_k_list=[]
#创建一个列表来存储行直线的截距
col_b_list=[]
#创建一个列表来存储行直线的斜率
row_k_list=[]
#创建一个列表来存储行直线的截距
row_b_list=[]
points_for_col_regression = []  # 用于存储列直线拟合点坐标列表
coordinates_for_col_regression = []  # 用于存储列直线拟合矩形坐标列表
points_for_row_regression=[]  # 用于存储行直线拟合点坐标列表
Matrix=[] #用于存储行列对应坐标矩阵，Matrix[i][j]表示第i+1行第j+1列的中心坐标
points_to_matrix_dict={} #用于存储点坐标到矩阵坐标的映射字典,points_to_matrix_dict[M[i][j]]表示第i+1行第j+1列点的行列坐标


#以下是定义的函数

#从points列表中删除plots列表中的点，返回剩余的点坐标构成的列表
def remove_plots_from_points(points, plots):
    # 遍历 plots 列表中的每个元素
    for plot in plots:
        # 尝试从 points 列表中移除当前 plot
        try:
            points.remove(plot)
        except ValueError:
            # 如果 plot 不在 points 列表中，忽略该异常
            pass
    return points


#判断两个矩形是否有重叠部分，通过计算相交面积占比判断是否相交，并返回true or false，以及所认为的预测错误的矩形（目前经验值，缺乏进一步测试验证）
def are_rectangles_intersecting(rect1, rect2,max_intersection_area_ratio):
    # rect1 和 rect2 分别为两个矩形，每个矩形表示为 ((x1, y1), (x2, y2))，其中 (x1, y1) 是左上角坐标，(x2, y2) 是右下角坐标
    # 提取矩形的坐标信息
    (x1_r1, y1_r1), (x2_r1, y2_r1) = rect1
    (x1_r2, y1_r2), (x2_r2, y2_r2)= rect2
    
    # 检查是否有重叠部分
    if (max(x1_r1, x1_r2) <= min(x2_r1, x2_r2) and max(y1_r1, y1_r2) <= min(y2_r1, y2_r2)):
        #绘制两个矩形
        #rect1_patch = patches.Rectangle(rect1[0], rect1[1][0]-rect1[0][0], rect1[1][1]-rect1[0][1], linewidth=2, edgecolor='green', facecolor='none')
        #rect2_patch = patches.Rectangle(rect2[0], rect2[1][0]-rect2[0][0], rect2[1][1]-rect2[0][1], linewidth=2, edgecolor='orange', facecolor='none')
        #ax.add_patch(rect1_patch)
        #ax.add_patch(rect2_patch)
        # 有重叠部分,计算相交面积
        intersection_x1 = max(x1_r1, x1_r2)
        intersection_y1 = max(y1_r1, y1_r2)
        intersection_x2 = min(x2_r1, x2_r2)
        intersection_y2 = min(y2_r1, y2_r2)
        print("相交矩形：", ((intersection_x1, intersection_y1), (intersection_x2, intersection_y2)))
        #intersection_patch = patches.Rectangle((intersection_x1, intersection_y1), intersection_x2-intersection_x1, intersection_y2-intersection_y1, linewidth=0, color='b')
        #ax.add_patch(intersection_patch)
        intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
        # 计算两个矩形的面积
        area1 = (x2_r1 - x1_r1) * (y2_r1 - y1_r1)
        area2 = (x2_r2 - x1_r2) * (y2_r2 - y1_r2)
        #print("矩形1面积：", area1)
        #print("矩形2面积：", area2)
        #print("相交面积：", intersection_area)
        # 计算相交比例
        intersection_ratio = intersection_area / (area1 + area2 - intersection_area)
        #print("相交面积占比：", intersection_ratio)
        intersection_ratio =intersection_ratio*(intersection_y2/img_size[1])
        print("相交面积占比(乘以了相交部分矩形的ymax/img_size[1])：", intersection_ratio)
        # 相交面积大于max_intersection_area_ratio时认为相交
        if intersection_ratio > max_intersection_area_ratio:
            print("相交面积占比大于阈值,为",intersection_ratio)
            if area1 > area2:
                #print("矩形1更大！")
                print("去除的矩形是：", rect1)
                remove_rect=patches.Rectangle(rect1[0], rect1[1][0]-rect1[0][0], rect1[1][1]-rect1[0][1], linewidth=5, edgecolor='red', facecolor='none')
                #ax.add_patch(remove_rect)
                ax.text(rect1[0][0], rect1[1][1], str(round(intersection_ratio, 4)), color='red', fontsize=10)
                return 1, rect1
            else:
                #print("矩形2更大！")
                print("去除的矩形是：", rect2)
                remove_rect=patches.Rectangle(rect2[0], rect2[1][0]-rect2[0][0], rect2[1][1]-rect2[0][1], linewidth=5, edgecolor='red', facecolor='none')
                #ax.add_patch(remove_rect)
                ax.text(rect2[0][0], rect2[1][1], str(round(intersection_ratio, 4)), color='red', fontsize=10)
                return 1, rect2
        else:
            return 0, None
    else:
        # 没有重叠部分
        return 0, None


#计算与给定点距离最近的某点,并返回该点的坐标(利用中心坐标)（注意points是坐标列表）
def find_nearest_point(point, points):
    min_distance = float('inf')  # 设置一个初始的最小距离，设为无穷大
    nearest_point = None  # 初始化最近点的坐标为None
    for p in points:
        # 计算点到给定点的距离
        distance = math.sqrt((p[0] - point[0])**2 + (p[1] - point[1])**2)
        # 如果当前点的距离比最小距离小，则更新最小距离和最近点的坐标
        if (distance < min_distance) and (distance!= 0):
            min_distance = distance
            nearest_point = p
    return nearest_point # 返回距离给定点最近点的坐标


#从指定的x坐标位置开始搜索，返回距离给定点最近的点坐标
#position参数用来指定搜索的范围（position=-1表示搜索x坐标小于给定点的点，position=1表示搜索x坐标大于给定点的点，position=0表示搜索所有点）
def find_nearest_point_by_x(point, points, position=0):
    min_distance = float('inf')  # 设置一个初始的最小距离，设为无穷大
    nearest_point = point  # 初始化最近点的坐标为point
    # 根据 position 参数的值来决定搜索空间
    if position == -1:
        # 只考虑 x 坐标小于给定点的情况
        filtered_points = [p for p in points if p[0] < point[0]]
    elif position == 1:
        # 只考虑 x 坐标大于给定点的情况
        filtered_points = [p for p in points if p[0] > point[0]]
    else:
        # 考虑所有点
        filtered_points = points
    for p in filtered_points:
        # 计算点到给定点的距离
        distance = math.sqrt((p[0] - point[0])**2 + (p[1] - point[1])**2)
        # 如果当前点的距离比最小距离小，则更新最小距离和最近点的坐标
        if distance < min_distance and distance != 0:
            min_distance = distance
            nearest_point = p
    return nearest_point


#找到中心坐标中第n大的y坐标对应的点坐标
def find_nth_largest_y_coordinate(points, n):
    # 如果点的数量少于n个，则无法找到第n大的y坐标
    if len(points) < n:
        return None
    # 使用一个集合来存储y坐标，以便找到第n大的y值
    unique_y_coordinates = set()
    for point in points:
        unique_y_coordinates.add(point[1])
    # 将y坐标排序，并找到第n大的值
    sorted_y_coordinates = sorted(unique_y_coordinates, reverse=True)
    nth_largest_y = sorted_y_coordinates[n - 1]
    # 找到第n大的y值对应的x坐标
    for point in points:
        if point[1] == nth_largest_y:
            return point


#从给定的矩形中心坐标中找到矩形右下角顶点坐标y值最大的点坐标，并返回该顶点对应的中心坐标
def find_nth_largest_y_coordinate_vertex(points, n,positoin=0):
    # 如果点的数量少于n个，则无法找到第n大的y坐标
    if len(points) < n:
        return None
    # 使用一个集合来存储矩形坐标对，以便找到第n大的y值
    unique_rect_coordinates = set()
    for point in points:
        unique_rect_coordinates.add(center_to_rectangle[point])
    # 将矩形坐标对按右下角坐标的y值排序（从小到大），并找到第n大的值
    sorted_rect_coordinates = sorted(unique_rect_coordinates, key=lambda x: x[1][1],reverse=True)
    #如果最大值有重复，则取对应的x更大（position=1时）或更小（position=-1时）的点
    if sorted_rect_coordinates[n - 1][1][1] == sorted_rect_coordinates[n-2][1][1]:
        if positoin==1:
            if sorted_rect_coordinates[n - 1][1][0] > sorted_rect_coordinates[n-2][1][0]:
                y_max_point=rectangle_to_center[sorted_rect_coordinates[n - 1]]
            else:
                y_max_point=rectangle_to_center[sorted_rect_coordinates[n-2]]
        if positoin==-1:
            if sorted_rect_coordinates[n - 1][1][0] < sorted_rect_coordinates[n-2][1][0]:
                y_max_point=rectangle_to_center[sorted_rect_coordinates[n - 1]] 
            else:
                y_max_point=rectangle_to_center[sorted_rect_coordinates[n-2]]
        else:
            y_max_point=rectangle_to_center[sorted_rect_coordinates[n - 1]]
    else:
        y_max_point=rectangle_to_center[sorted_rect_coordinates[n - 1]]
    return y_max_point


#依据给定的摄像头位置position,寻找距离给定中心坐标对应的矩形顶点最近的矩形顶点，进而找到用于拟合的下一个中心坐标
#position参数用来指定搜索的范围（position=-1表示搜索x坐标小于给定点的点，position=1表示搜索x坐标大于给定点的点，position=0表示搜索所有点）
def find_nearest_point_by_rectangulat_vertex(point, points,position=0):
    min_distance = float('inf')  # 设置一个初始的最小距离，设为无穷大
    nearest_point = point  # 初始化最近点的坐标为point
    filtered_points=[]
    # 根据 position 参数的值来决定搜索空间
    if position == -1:
        # 只考虑 x 坐标小于给定点的情况
        rectangle_vertex_now=(center_to_rectangle[point][0][0],center_to_rectangle[point][0][1]) #左侧时获取给定点的矩形的左上角坐标
        rectangles_coordiantes=[] #存储矩形坐标对
        for p in points:
            if p[0] < point[0]:
                filtered_points.append(p) #只搜寻小于给定点x坐标的点
                rectangles_coordiantes.append(center_to_rectangle[p]) #存储对应搜寻点的矩形坐标对
        # 遍历所有矩形右下角顶点坐标找到距离最近的矩形
        for rectangle_vertex in rectangles_coordiantes:
            rectangle_vertex_right_bottom=(rectangle_vertex[1][0],rectangle_vertex[1][1])
            distance = math.sqrt((rectangle_vertex_right_bottom[0] - rectangle_vertex_now[0])**2 + (rectangle_vertex_right_bottom[1] - rectangle_vertex_now[1])**2)
            if distance < min_distance and distance != 0:
                min_distance = distance
                nearest_point = rectangle_to_center[rectangle_vertex] #更新最近点的坐标为距离该点最近的矩形的中心坐标
    elif position == 1:
        # 只考虑 x 坐标大于给定点的情况
        rectangle_vertex_now=(center_to_rectangle[point][1][0],center_to_rectangle[point][0][1]) #右侧时获取给定点的矩形的右上角坐标
        rectangles_coordiantes=[] #存储矩形坐标对
        for p in points:
            if p[0] > point[0]:
                filtered_points.append(p) #只搜寻大于给定点x坐标的点
                rectangles_coordiantes.append(center_to_rectangle[p]) #存储对应搜寻点的矩形坐标对
        # 遍历所有矩形左下角顶点坐标找到距离最近的矩形
        for rectangle_vertex in rectangles_coordiantes:
            rectangle_vertex_left_bottom=(rectangle_vertex[0][0],rectangle_vertex[0][1])
            distance = math.sqrt((rectangle_vertex_left_bottom[0] - rectangle_vertex_now[0])**2 + (rectangle_vertex_left_bottom[1] - rectangle_vertex_now[1])**2)
            if distance < min_distance and distance != 0:
                min_distance = distance
                nearest_point = rectangle_to_center[rectangle_vertex] #更新最近点的坐标为距离该点最近的矩形的中心坐标
    else:
        # 考虑所有点
        filtered_points = points # 存储所有点坐标
        # 遍历所有中心坐标找到距离最近的矩形
        for p in points:
            distance = math.sqrt((p[0] - point[0])**2 + (p[1] - point[1])**2)
            if distance < min_distance and distance != 0:
                min_distance = distance
                nearest_point = p #更新最近点的坐标为距离该点最近的中心坐标
    #print("最近点坐标：",nearest_point)
    return nearest_point


#对points里的坐标进行直线拟合，返回斜率和截距(注意points要是numpy数组)
def linear_regression(points):
    [vx, vy, x, y] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    # 计算斜率和截距
    k= vy / vx
    b = y - k * x
    return k, b # 返回斜率和截距


#判断直线（y=kx+b）是否与矩形（左上角坐标为p[0]，右下角坐标为p[1]）相交，返回True或False
def is_line_intersect_rectangle(k, b, p):
    # 提取矩形坐标
    if k==0:
        col_flag=False #是否为列直线
        draw_center_point_and_rect(center_xy,ax,img_size) #绘制中心点和矩形框
        set_axis_and_save(col_flag,ax,img_name,img_size,save_dir)#设置坐标轴范围和保存路径等
        exit() #退出程序 
    assert k!= 0, "斜率不能为0"
    rect_x1, rect_y1 = p[0][0], p[0][1]  # 左上角坐标
    rect_x2, rect_y2 = p[1][0], p[1][1]  # 右下角坐标
    # 检查左边界 x = rect_x1
    y_left = k * rect_x1 + b
    if rect_y1 <= y_left <= rect_y2:
        return True
    # 检查右边界 x = rect_x2
    y_right = k * rect_x2 + b
    if rect_y1 <= y_right <= rect_y2:
        return True
    # 检查上边界 y = rect_y2
    x_top = (rect_y2 - b) / k
    if rect_x1 <= x_top <= rect_x2:
        return True
    # 检查下边界 y = rect_y1
    x_bottom = (rect_y1 - b) / k
    if rect_x1 <= x_bottom <= rect_x2:
        return True
    return False


#定义绘制直线函数
def draw_line(k,b,ax,c,begin_x,end_x):
    x = np.array([begin_x, end_x])
    y = k * x + b
    ax.plot(x, y, color=c, linewidth=2)


#拟合列直线（距离摄像头较近位置，如一二列）
def loop_col_regression(first_point,points,ax):
    #print("第1个点坐标：",first_point)
    #定义一个空列表,用于存储用于拟合的点
    points_for_regression = []
    #定义一个空列表,用于存储用于拟合的矩形框坐标对
    rectangles_for_regression = []
    #存入拟合起始点坐标
    points_for_regression.append(tuple(first_point))
    #存入拟合起始矩形坐标对
    rectangles_for_regression.append(center_to_rectangle[tuple(first_point)])
    #center_xy_copy = list(remove_plots_from_points(points,first_point)) #从中心坐标列表中删除拟合过的坐标
    # 找到与第一个中心点距离最近的点
    point_now = find_nearest_point(first_point, remove_plots_from_points(points,points_for_regression))
    #center_xy_copy = list(remove_plots_from_points(points,point_now)) #从中心坐标列表中删除拟合过的坐标
    #print("第2个点坐标：",point_now)
    #存入当前的拟合点坐标
    points_for_regression.append(point_now)
    #存入当前的拟合矩形坐标对
    rectangles_for_regression.append(center_to_rectangle[point_now])
    i=1
    #print("第",i,"次拟合完成")
    while i<nums_of_row:
        #对存好的点进行直线拟合
        k, b = linear_regression(np.array(points_for_regression))
        #找到与当前点最近的点的坐标
        point_next = find_nearest_point(point_now, remove_plots_from_points(points,points_for_regression))
        #print("第",i,"次拟合点坐标：",point_next)
        #判断拟合的直线与下一个矩形是否相交
        if is_line_intersect_rectangle(k, b, center_to_rectangle[point_next]):
            #相交，则将该点坐标加入到拟合点中
            points_for_regression.append(point_next)
            #存入当前的拟合矩形坐标对
            rectangles_for_regression.append(center_to_rectangle[point_next])
            #center_xy_copy = list(remove_plots_from_points(points,point_next)) #从中心坐标列表中删除拟合过的坐标
            #继续进行直线拟合
            i+=1
            #更新当前点坐标
            point_now = point_next
        #    print("第",i+2,"个点坐标：",point_now) 
        #    print("第",i,"次拟合完成")
        else:
            #不相交,该列拟合完成
            #print("不相交")
            
            break
    #draw_line(k,b,ax,'green') #绘制拟合的直线
    col_k_list.append(k) #记录斜率
    col_b_list.append(b) #记录截距
    #points=list(remove_plots_from_points(points,points_for_regression)) #从中心坐标列表中删除拟合过的坐标
    #points=list(remove_plots_from_points(points,[first_point])) #从中心坐标列表中删除第一个坐标
    return points_for_regression,rectangles_for_regression #返回拟合的点坐标和拟合的矩形坐标对


#拟合列直线(距离摄像头较远位置，如三四五列)(目前已经不使用了)
def loop_col_regression_by_x(first_point,points,ax,position): 
    #print("第1个点坐标：",first_point)
    #定义一个空列表,用于存储用于拟合的点
    points_for_regression = []
    #存入拟合起始点坐标
    points_for_regression.append(tuple(first_point))
    # 找到与第一个中心点距离最近的点
    point_now = find_nearest_point_by_x(tuple(first_point), remove_plots_from_points(points,points_for_regression),position)
    #print("第2个点坐标：",point_now)
    #存入当前的拟合点坐标
    points_for_regression.append(point_now)
    i=1
    #print("第",i,"次拟合完成")
    while i<6:
        #对存好的点进行直线拟合
        k, b = linear_regression(np.array(points_for_regression))
        #print("斜率：",k,"截距：",b)
        #找到与当前点最近的点的坐标
        point_next = find_nearest_point_by_x(point_now,remove_plots_from_points(points,points_for_regression),position)
        print("第",i,"次拟合点坐标：",point_next)
        #判断拟合的直线与下一个矩形是否相交
        if is_line_intersect_rectangle(k, b, center_to_rectangle[point_next]):
            #继续进行直线拟合
            i+=1
            #更新当前点坐标
            if point_next!=point_now :#相交且两点不是同一点，则将该点坐标加入到拟合点中
                points_for_regression.append(point_next)
                point_now = point_next
                print("第",i+1,"个点坐标：",point_now) 
            print("第",i,"次拟合完成")
        else:
            #不相交,该列拟合完成
            print("不相交")
            print(i)
            break
    draw_line(k,b,ax,'green') #绘制拟合的直线
    #print("绘制直线的斜率和截距：",k,b)
    return points_for_regression


#找出剩余的矩形坐标对
def function_coordinates_rectangles_remain(center_xy_copy):
    coordinates_copy = []#创建一个空列表，用于存储剩余的矩形坐标对
    for plot in center_xy_copy:
        coordinates_copy.append(center_to_rectangle[plot])
    return coordinates_copy


#判断给定斜率和截距的直线y=kx+b是否与任意个给定的矩形框coordinates（列表，元素为元组坐标对（包含左上角坐标(x1,y1)和右下角坐标(x2,y2)））相交，
#如果相交，返回True和相交矩形的中心坐标列表，否则返回False。
#position表示摄像头位置，-1表示左侧，1表示右侧，0表示不限制位置(-1表示只从起点的左侧寻找矩形，1表示只从起点的右侧寻找矩形，0表示不限制位置)
#begin_point表示直线的起点坐标
def line_rect_intersection(k,b,coordinates,begin_point,position):
    flag=False
    center_point=[] #相交矩形的中心坐标
    if k==0:
        col_flag=False #是否为列直线
        draw_center_point_and_rect(center_xy,ax,img_size) #绘制中心点和矩形框
        set_axis_and_save(col_flag,ax,img_name,img_size,save_dir)#设置坐标轴范围和保存路径等
        exit() #退出程序
    assert k!= 0, "斜率不能为0"
    for plot in coordinates :
        center_xy_plot=(plot[0][0]+plot[1][0])/2, (plot[0][1]+plot[1][1])/2 #矩形中心坐标
        if position==-1:
            if center_xy_plot[0]<begin_point[0] :
                (x1,y1),(x2,y2) = plot #提取矩形坐标
                #检查左边界
                x=x1
                y_left=k*x+b
                if y1<=y_left<=y2:
                    flag=True 
                    center_point.append(center_xy_plot)  #直线与矩形相交
                    continue
                #检查右边界
                x=x2
                y_right=k*x+b
                if y1<=y_right<=y2:
                    flag=True
                    center_point.append(center_xy_plot)  #直线与矩形相交
                    continue
                #检查上边界
                y=y1
                x_up=(-b-y)/k
                if x1<=x_up<=x2:
                    flag=True
                    center_point.append(center_xy_plot)  #直线与矩形相交
                    continue
                #检查下边界
                y=y2
                x_down=(-b-y)/k
                if x1<=x_down<=x2:
                    flag=True
                    center_point.append(center_xy_plot)  #直线与矩形相交
                    continue
                #检查上边界
                y=y1
                x_top=(y-b)/k
                if x1<=x_top<=x2:
                    flag=True
                    center_point.append(center_xy_plot)  #直线与矩形相交
                    continue
                #检查下边界
                y=y2
                x_bottom=(y-b)/k
                if x1<=x_bottom<=x2:
                    flag=True      
                    center_point.append(center_xy_plot)  #直线与矩形相交
                    continue
            else:
                continue
        elif position==1:
            if center_xy_plot[0]>begin_point[0] :
                (x1,y1),(x2,y2) = plot #提取矩形坐标
                
                #检查左边界
                x=x1
                y_left=k*x+b
                if y1<=y_left<=y2:
                    flag=True 
                    center_point.append(center_xy_plot)  #直线与矩形相交
                    continue
                #检查右边界
                x=x2
                y_right=k*x+b
                if y1<=y_right<=y2:
                    flag=True
                    center_point.append(center_xy_plot)  #直线与矩形相交
                    continue
                #检查上边界
                y=y1
                x_up=(-b-y)/k
                if x1<=x_up<=x2:
                    flag=True
                    center_point.append(center_xy_plot)  #直线与矩形相交
                    continue
                #检查下边界
                y=y2
                x_down=(-b-y)/k
                if x1<=x_down<=x2:
                    flag=True
                    center_point.append(center_xy_plot)  #直线与矩形相交
                    continue
                #检查上边界
                y=y1
                x_top=(y-b)/k
                if x1<=x_top<=x2:
                    flag=True
                    center_point.append(center_xy_plot)  #直线与矩形相交
                    continue
                #检查下边界
                y=y2
                x_bottom=(y-b)/k
                if x1<=x_bottom<=x2:
                    flag=True      
                    center_point.append(center_xy_plot)  #直线与矩形相交
                    continue
            else:
                continue
        else:
            if begin_point!=center_xy_plot: #排除起点坐标
                (x1,y1),(x2,y2) = plot #提取矩形坐标
                
                #检查左边界
                x=x1
                y_left=k*x+b
                if y1<=y_left<=y2:
                    flag=True 
                    center_point.append(center_xy_plot)  #直线与矩形相交
                    continue
                #检查右边界
                x=x2
                y_right=k*x+b
                if y1<=y_right<=y2:
                    flag=True
                    center_point.append(center_xy_plot)  #直线与矩形相交
                    continue
                #检查上边界
                y=y1
                x_up=(-b-y)/k
                if x1<=x_up<=x2:
                    flag=True
                    center_point.append(center_xy_plot)  #直线与矩形相交
                    continue
                #检查下边界
                y=y2
                x_down=(-b-y)/k
                if x1<=x_down<=x2:
                    flag=True
                    center_point.append(center_xy_plot)  #直线与矩形相交
                    continue
                #检查上边界
                y=y1
                x_top=(y-b)/k
                if x1<=x_top<=x2:
                    flag=True
                    center_point.append(center_xy_plot)  #直线与矩形相交
                    continue
                #检查下边界
                y=y2
                x_bottom=(y-b)/k
                if x1<=x_bottom<=x2:
                    flag=True      
                    center_point.append(center_xy_plot)   #直线与矩形相交
                    continue
            else:#直线与矩形框不相交
                continue                        
    return flag,center_point    


#判断给定斜率和截距的直线y=kx+b是否与任意个给定的矩形框coordinates（列表，元素为元组坐标对（包含左上角坐标(x1,y1)和右下角坐标(x2,y2)））相交，
#如果相交，返回True和相交矩形的中心坐标列表，否则返回False。
#当前不在使用
def line_rect_intersection_no_first_point(k,b,coordinates):
    flag=False
    center_point=[] #相交矩形的中心坐标    
    for plot in coordinates :
        center_xy_plot=(plot[0][0]+plot[1][0])/2, (plot[0][1]+plot[1][1])/2 #矩形中心坐标
        (x1,y1),(x2,y2) = plot #提取矩形坐标
        #检查左边界
        x=x1
        y_left=k*x+b
        if y1<=y_left<=y2:
            flag=True 
            center_point.append(center_xy_plot)  #直线与矩形相交
            continue
        #检查右边界
        x=x2
        y_right=k*x+b
        if y1<=y_right<=y2:
            flag=True
            center_point.append(center_xy_plot)  #直线与矩形相交
            continue
        #检查上边界
        y=y1
        x_up=(-b-y)/k
        if x1<=x_up<=x2:
            flag=True
            center_point.append(center_xy_plot)  #直线与矩形相交
            continue
        #检查下边界
        y=y2
        x_down=(-b-y)/k
        if x1<=x_down<=x2:
            flag=True
            center_point.append(center_xy_plot)  #直线与矩形相交
            continue
        #检查上边界
        y=y1
        x_top=(y-b)/k
        if x1<=x_top<=x2:
            flag=True
            center_point.append(center_xy_plot)  #直线与矩形相交
            continue
        #检查下边界
        y=y2
        x_bottom=(y-b)/k
        if x1<=x_bottom<=x2:
            flag=True      
            center_point.append(center_xy_plot)  #直线与矩形相交
            continue
    return flag,center_point    



#判断给定斜率和截距的直线y=kx+b是否与任意个给定的矩形框coordinates（列表，元素为元组坐标对（包含左上角坐标(x1,y1)和右下角坐标(x2,y2)））相交，
#如果相交，返回True和相交矩形的中心坐标列表，否则返回False。
#当前正在使用
def check_line_intersects_rectangles(k, b, coordinates):
    def does_line_intersect_rect(k, b, x1, y1, x2, y2):
        if x1 > x2: x1, x2 = x2, x1
        if y1 < y2: y1, y2 = y2, y1
        # Calculate intersection points with the four edges of the rectangle
        # Left edge (x1, y)
        y_left = k * x1 + b
        if y2 <= y_left <= y1:
            return True
        # Right edge (x2, y)
        y_right = k * x2 + b
        if y2 <= y_right <= y1:
            return True
        # Top edge (x, y1)
        if k != 0:
            x_top = (y1 - b) / k
            if x1 <= x_top <= x2:
                return True
        # Bottom edge (x, y2)
        if k != 0:
            x_bottom = (y2 - b) / k
            if x1 <= x_bottom <= x2:
                return True
        return False
    #计算矩形中心坐标
    def get_center_of_rect(x1, y1, x2, y2):
        if x1 > x2: x1, x2 = x2, x1
        if y1 < y2: y1, y2 = y2, y1
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    intersecting_centers = []
    for rect in coordinates:
        (x1, y1), (x2, y2) = rect
        if does_line_intersect_rect(k, b, x1, y1, x2, y2):
            intersecting_centers.append(get_center_of_rect(x1, y1, x2, y2))
    if intersecting_centers:
        return True, intersecting_centers
    else:
        return False, []


#拟合列直线(best) （目前正在使用）
def loop_col_regression_best(first_point,points,ax,position): 
    print("第1个点坐标：",first_point)
    #定义一个空列表,用于存储用于拟合的点
    points_for_regression = []
    #定义一个空列表，用于存储用于拟合的矩形坐标对
    rectangles_for_regression = []
    #存入拟合起始点坐标
    #存入拟合起始点坐标
    points_for_regression.append(tuple(first_point))
    #存入拟合起始矩形坐标对
    rectangles_for_regression.append(center_to_rectangle[tuple(first_point)])
    # 找到与第一个中心点距离最近的点
    point_now = find_nearest_point_by_rectangulat_vertex(tuple(first_point), remove_plots_from_points(points,points_for_regression),position)
    print("可能的第2个点坐标：",point_now)
    error_2rd_point1=point_now
    k, b = linear_regression(np.array([first_point,point_now]))#对第一个点和第二个点进行直线拟合
    k_flag=True
    if len(col_k_list)>1:    
        col_k_avg=sum(col_k_list[1:len(col_k_list)])/(len(col_k_list)-1)
        k_relative_error=abs((k-col_k_avg)/col_k_avg)*100
        print("斜率相对误差：",k_relative_error,"%")
        if k_relative_error>col_slope_error_threshold:
            k_flag=False
    is_line_intersect_rectangle_flag=0
    if k_flag:
        for p in center_xy:
            if p!=tuple(first_point) and p!=error_2rd_point1 :
                if is_line_intersect_rectangle(k, b, center_to_rectangle[p])==1:
                    print("第一个点与第二个点的斜率与前列斜率的相对误差小于阈值，并且至少穿过一个矩形，说明第二个点寻找正确")
                    points_for_regression.append(point_now)
                    points=remove_plots_from_points(points,points_for_regression)
                    print("第",2,"个点坐标：",point_now)
                    rectangles_for_regression.append(center_to_rectangle[point_now])
                    is_line_intersect_rectangle_flag=1
                    break
    if is_line_intersect_rectangle_flag==0 or k_flag==False:    
        if k_flag: print("不穿过任何矩形,重新寻找")
        if k_flag==False:
            print("不穿过任何矩形，而且斜率相对误差大于阈值，重新寻找")
        points_for_regression_cp=[tuple(first_point),point_now]
        #print(points_for_regression_cp)
        points_copy=points.copy()
        points_copy=list(remove_plots_from_points(points_copy,points_for_regression_cp))
        #print(points_copy)
        point_now=find_nearest_point_by_rectangulat_vertex(tuple(first_point), points_copy,position)
        print("重新寻找的第2个点坐标：",point_now)
        error_2rd_point2=point_now
        k,b = linear_regression(np.array([first_point,point_now]))#对第一个点和第二个点进行直线拟合
        k_flag=True
        if len(col_k_list)>1:    
            col_k_avg=sum(col_k_list[1:len(col_k_list)])/(len(col_k_list)-1)
            k_relative_error=abs((k-col_k_avg)/col_k_avg)*100
            print("斜率相对误差：",k_relative_error,"%")
            if k_relative_error>col_slope_error_threshold:
                points_for_regression_cp.append(error_2rd_point2)
                k_flag=False
        is_line_intersect_rectangle_flag=0
        for p in center_xy:
            if p!=tuple(first_point) and p!=error_2rd_point1 and p!=error_2rd_point2:
                if is_line_intersect_rectangle(k, b, center_to_rectangle[p])==1 and k_flag:
                    print("第一个点与第二个点的斜率与前列斜率的相对误差小于阈值，并且至少穿过一个矩形，说明第二个点寻找正确")
                    #points_for_regression.append(error_2rd_point2)
                    #points=remove_plots_from_points(points,points_for_regression)
                    print("第",2,"个点坐标：",error_2rd_point2)
                    #rectangles_for_regression.append(center_to_rectangle[error_2rd_point2])
                    is_line_intersect_rectangle_flag=1
                    break
        if is_line_intersect_rectangle_flag==0 or k_flag==False:    
            if k_flag: print("不穿过任何矩形,重新寻找")
            if k_flag==False:
                print("不穿过任何矩形，而且斜率相对误差大于阈值，重新寻找")
            #print(points_for_regression_cp)
            points_copy=points.copy()
            points_copy=list(remove_plots_from_points(points_copy,points_for_regression_cp))
            #print(points_copy)
            point_now=find_nearest_point_by_rectangulat_vertex(tuple(first_point), points_copy,position)
            print("重新寻找的第2个点坐标：",point_now)
        points_for_regression.append(point_now)
        rectangles_for_regression.append(center_to_rectangle[point_now])
        points=remove_plots_from_points(points,points_for_regression)
        #print(points_for_regression)
        #print(points)
        #存入当前的拟合点坐标
    #points_for_regression.append(point_now)
    #存入当前的拟合矩形坐标对
    #rectangles_for_regression.append(center_to_rectangle[point_now])
    i=1
    #print("第",i,"次拟合完成")
    while i<nums_of_row:
        #对存好的点进行直线拟合
        k, b = linear_regression(np.array(points_for_regression))
        print("斜率：",k,"截距：",b)
        #找到与当前点最近的点的坐标
        point_next = find_nearest_point_by_rectangulat_vertex(point_now,remove_plots_from_points(points,points_for_regression),position)
        print("第",i,"次拟合点坐标：",point_next)
        #判断拟合的直线与下一个矩形是否相交
        if is_line_intersect_rectangle(k, b, center_to_rectangle[point_next]):
            #print("第",i,"次拟合直线与最近点矩形相交")
            #继续进行直线拟合
            i+=1
            #更新当前点坐标
            if point_next!=point_now :#相交且两点不是同一点，则将该点坐标加入到拟合点中
                points_for_regression.append(point_next)
                rectangles_for_regression.append(center_to_rectangle[point_next])
                point_now = point_next
                print("第",i+1,"个点坐标：",point_now) 
            #print("第",i,"次拟合完成")
        else:
            #print("point_now:",point_now)
            #print("points:",points)
            flag,center_point=line_rect_intersection(k, b, function_coordinates_rectangles_remain(points),point_now,position) #判断直线与矩形框是否相交,返回相交矩形的中心坐标列表
            if flag:
                print("相交")
                print("相交矩形的中心坐标为：",center_point)
                nearest_centerpoint=find_nearest_point_by_rectangulat_vertex(point_now,center_point,position) #找到距离直线起点最近的矩形中心坐标
                print("第",i,"次拟合相交矩形的中心坐标为：",nearest_centerpoint)
                points_for_regression.append(nearest_centerpoint) #将距离最近的相交矩形的中心坐标加入到拟合点中
                rectangles_for_regression.append(center_to_rectangle[nearest_centerpoint]) #将距离最近的相交矩形的中心坐标对应的矩形坐标对加入到拟合矩形坐标对中
                continue
            else:
                if len(points_for_regression)==2:
                    print("不相交且拟合点数小于等于3，重新寻找第二个点")
                    points_copy=points.copy()
                    points_for_regression_cp=points_for_regression.copy()
                    points_copy=list(remove_plots_from_points(points_copy,points_for_regression_cp))
                    #print(points_copy)
                    point_now=find_nearest_point_by_rectangulat_vertex(tuple(first_point), points_copy,position)
                    print("重新寻找的第2个点坐标：",point_now)
                    points_for_regression.append(point_now)
                    rectangles_for_regression.append(center_to_rectangle[point_now])
                    points=remove_plots_from_points(points,points_for_regression)
                    #print(points_for_regression)
                    continue
                else:
                    print("不相交")
                    break
    #draw_line(k,b,ax,'green') #绘制拟合的直线
    col_k_list.append(k)#记录斜率
    col_b_list.append(b)#记录截距
    #print("绘制直线的斜率和截距：",k,b)
    return points_for_regression,rectangles_for_regression #返回拟合点坐标列表和拟合矩形坐标对列表


#在列直线拟合正常后使用该函数拟合行直线
#拟合行直线,返回斜率和截距
#colpoints表示列坐标列表，每个元素为列表，包含该列的中心点坐标。
def loop_row_regression(colpoints):
    k, b = linear_regression(np.array(colpoints))
    return k, b #返回斜率，截距


#创建一个矩阵Matrix，Matrix[i][j]表示第i+1行第j+1列的中心坐标
#position表示摄像头位置，-1表示左侧，1表示右侧
#position==-1时，列数自然从左侧开始
#position==1时，列数从右侧开始，所以需要反转一下
def construct_coordinate_matrix(points_for_row_regression, points_for_col_regression,position):
    num_rows = len(points_for_row_regression)
    num_cols = len(points_for_col_regression)
    # 初始化矩阵
    Matrix = [[None] * num_cols for _ in range(num_rows)]
    # 填充矩阵
    if position == 1: #摄像头在右侧，反转列数（colnum-1-i）
        for i in range(num_rows):
            for j in range(num_cols):
                Matrix[i][j] = (points_for_row_regression[i][num_cols-1-j][0], points_for_col_regression[num_cols-1-j][i][1])
    else :  #摄像头在左侧或不限制位置，列数按自然顺序（从左侧开始）
        for i in range(num_rows):
            for j in range(num_cols):
                Matrix[i][j] = (points_for_row_regression[i][j][0], points_for_col_regression[j][i][1])
    return Matrix


#创建一个字典，将中心坐标映射到所在的行与列
#函数输入：中心坐标矩阵Matrix
#函数输出：中心坐标-行列字典points_to_matrix_dict
def create_coordinate_dictionary(Matrix):
    points_to_matrix_dict = {}
    cols=len(Matrix[0])
    rows=len(Matrix)
    for i in range(rows):
        for j in range(cols):
            points_to_matrix_dict[Matrix[i][j]] = (i+1, j+1)
    return points_to_matrix_dict


#对输入一组坐标元组，按坐标的y值从大到小排序，
# 并输出排序后的坐标元组。
def sort_points_by_ymax_to_ymin(points):
    # 按y值从大到小排序
    sorted_points = sorted(points, key=lambda x: x[1], reverse=True)
    return sorted_points


#读取整个文件夹内中图片进行拟合时用来关闭程序的函数
def close_plot():
    plt.close()
    sys.exit()


#不再使用
#判断摄像头位置，返回摄像头位置和第一个中心坐标(中心坐标y值最大的点)
def judge_camera_position(points,img_size):
    # 将points转换为numpy数组并赋值给coordinates变量
    coordinates = np.array(points)
    # 使用 NumPy 的 argmax 函数找到具有最大 y 坐标的点的索引
    max_y_index = np.argmax(coordinates[:, 1])
    # 获取具有最大 y 坐标的点的坐标
    max_y_point = coordinates[max_y_index]
    first_point = max_y_point
    if max_y_point[0] < img_size[0] / 2:
        #print("摄像头在左侧")
        return -1,  tuple(first_point)
    else:
        #print("摄像头在右侧")
        return 1,   tuple(first_point)


#正在使用
#判断摄像头位置，返回摄像头位置和第一个中心坐标(矩形y值最大的点)
def judge_camera_position_best(points,img_size):
    flag, first_point = judge_camera_position(points,img_size)
    rect_y_max=center_to_rectangle[first_point][1][1]
    for point in points:
        if rect_y_max<center_to_rectangle[point][1][1]:
            rect_y_max=center_to_rectangle[point][1][1]
            first_point=point

    return flag,  tuple(first_point)



'''

            if flag==-1:
                if center_to_rectangle[point][0][0]<center_to_rectangle[first_point][0][0]:
                    rect_y_max=center_to_rectangle[point][1][1]
                    first_point=point
            if flag==1:
                if center_to_rectangle[point][1][0]>center_to_rectangle[first_point][1][0]:
                    rect_y_max=center_to_rectangle[point][1][1]
                    first_point=point

'''



#去除列表中重复的元素
def remove_duplicates(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


#判断两个矩形是否有重叠部分，并计算相交面积占比
#返回1表示有重叠部分，返回0表示无重叠部分，返回相交面积占比
def rect_inter_ratio(rect1,rect2):
    #提取矩形的坐标
    (x1_r1, y1_r1), (x2_r1, y2_r1) = rect1
    (x1_r2, y1_r2), (x2_r2, y2_r2) = rect2
    #检查是否有重叠部分
    if(max(x1_r1, x1_r2)<=min(x2_r1, x2_r2) and max(y1_r1, y1_r2)<=min(y2_r1, y2_r2)):#有重叠部分
        #计算相交面积
        intersection_x1=max(x1_r1, x1_r2)
        intersection_y1=max(y1_r1, y1_r2)
        intersection_x2=min(x2_r1, x2_r2)
        intersection_y2=min(y2_r1, y2_r2)
        print("相交矩形的坐标为：",(intersection_x1, intersection_y1),(intersection_x2, intersection_y2))
        #intersection_patch = patches.Rectangle((intersection_x1, intersection_y1), intersection_x2-intersection_x1, intersection_y2-intersection_y1, linewidth=0, color='b')
        #ax.add_patch(intersection_patch)
        intersection_area = (intersection_x2-intersection_x1)*(intersection_y2-intersection_y1)
        #计算两个矩形的面积
        area1 = (x2_r1-x1_r1)*(y2_r1-y1_r1)
        area2 = (x2_r2-x1_r2)*(y2_r2-y1_r2)
        #计算相交面积占比
        intersection_area_ratio = intersection_area/(area1+area2-intersection_area)
        #print("相交面积占比：",intersection_area_ratio)
        intersection_area_ratio =intersection_area_ratio*(intersection_y2/img_size[1])
        print("相交面积占比(乘以了相交部分矩形的ymax/img_size[1]):", intersection_area_ratio)
        return 1,intersection_area_ratio
    else:
        return 0,0.0


#找出一列矩形中两两相交矩形面积的最大对应的那两个矩形
#返回1表示找到了相交矩形，返回0表示没有找到相交矩形
#返回两个矩形的坐标max_rectangles[0],max_rectangles[1]，
#返回最大相交面积占比 max_area
def find_intersect_max_area_rectangles(rectangles):
    max_area=0
    rect1=[]
    rect2=[]
    f=0
    for i in range(len(rectangles)):
        if i==len(rectangles)-1:
            break
        for j in range(i+1,len(rectangles)):
            if j==len(rectangles)-1:
                break
            f,s=rect_inter_ratio(rectangles[i],rectangles[j])
            if f and s>max_area:
                max_area=s
                rect1=rectangles[i]
                rect2=rectangles[j]
                f=1
    return f,rect1,rect2,max_area



#判断两个矩形是否有重叠部分，通过计算相交面积占比判断是否相交，并返回true or false，以及两相交矩形的合并矩形（目前经验值，缺乏进一步测试验证）
#返回1表示找到了相交矩形，返回0表示没有找到相交矩形
#返回两相交矩形合并后的矩形，以及相交面积占比
def merge_rectangles(rect1, rect2,max_intersection_area_ratio):
    #提取矩形的坐标
    (x1_r1, y1_r1), (x2_r1, y2_r1) = rect1
    (x1_r2, y1_r2), (x2_r2, y2_r2) = rect2
    flag,intersection_area_ratio=rect_inter_ratio(rect1,rect2)
    if flag :
        #有重叠部分
        if intersection_area_ratio>max_intersection_area_ratio:
            print("相交面积占比大于阈值，为",intersection_area_ratio)
            print("进行合并")
            #计算合并后的矩形
            center_x = (x1_r1+x2_r1+x1_r2+x2_r2)/4
            center_y = (y1_r1+y2_r1+y1_r2+y2_r2)/4
            width = ((x2_r1-x1_r1)+(x2_r2-x1_r2))/2
            height = ((y2_r1-y1_r1)+(y2_r2-y1_r2))/2
            merged_rectangle = ((center_x-width/2, center_y-height/2),(center_x+width/2, center_y+height/2))
            ax.text(merged_rectangle[0][0],merged_rectangle[1][1],str(round(intersection_area_ratio,4)),color='red',fontsize=10)
            print("合并后的矩形坐标为：",merged_rectangle)
            #绘制合并后的矩形
            #merged_patch = patches.Rectangle(merged_rectangle[0], merged_rectangle[1][0]-merged_rectangle[0][0], merged_rectangle[1][1]-merged_rectangle[0][1], linewidth=5, edgecolor='green', facecolor='none')
            #ax.add_patch(merged_patch)
            return 1, merged_rectangle, intersection_area_ratio
        else:
            print("相交面积占比小于阈值，为",intersection_area_ratio)
            print("不进行合并")
            return 0, None,0.0
    else:
        print("无重叠部分")
        return 0, None,0.0


#传入一个坐标point和该坐标所在的列表points，根据其对应的矩形坐标对的最左侧的x坐标，进行排序，
#返回排序后的列表points
def sort_points_by_rect_left_x(point, points):
    #根据矩形坐标对的最右侧的x坐标进行排序
    coordinates=[]
    sorted_points=[]
    for i in range(len(points)):
        coordinates.append(center_to_rectangle[points[i]])
    coordinates.sort(key=lambda x: x[0][0])
    for i in range(len(coordinates)):
        sorted_points.append(rectangle_to_center[coordinates[i]])
    print("排序后的点坐标：", sorted_points)
    for i in range(len(sorted_points)):
        if point == sorted_points[i]:
            order=i
            break
    return order+1


#传入一个坐标point和该坐标所在的列表points，根据其对应的矩形坐标对的最右侧的x坐标，进行排序，
#返回排序后的列表points
def sort_points_by_rect_right_x(point, points):
    #根据矩形坐标对的最右侧的x坐标进行排序
    coordinates=[]
    sorted_points=[]
    for i in range(len(points)):
        coordinates.append(center_to_rectangle[points[i]])
    coordinates.sort(key=lambda x: x[1][0], reverse=True )
    for i in range(len(coordinates)):
        sorted_points.append(rectangle_to_center[coordinates[i]])
    print("排序后的点坐标：", sorted_points)
    for i in range(len(sorted_points)):
        if point == sorted_points[i]:
            order=i
            break
    return order+1


#判断找到的第一个点是否正确（也就是第一个点到底存不存在）
#传入找到的第一个点坐标first_point，以及所有矩形的中心坐标center_xy,摄像头位置position
def is_first_point_correct(first_point, center_xy, position):
    #根据摄像头位置分类判断
    if position == -1:  # 摄像头在左侧
        order = sort_points_by_rect_left_x(first_point, center_xy)
        print("第一个点的排序为zuo：", order)
        if  1<=order<=9:  # 第一个点在第一列
            return 1  #说明找到了正确的第一个点
        else:
            return 0  #说明找到了错误的第一个点
    if position == 1:  # 摄像头在右侧
        order = sort_points_by_rect_right_x(first_point, center_xy)
        print("第一个点的排序为right：", order)
        if 1<=order<=12:  # 第一个点在第五列
            return 1  #说明找到了正确的第一个点
        else:
            return 0  #说明找到了错误的第一个点


#定义拟合全部列直线的函数
#传入第一个中心坐标first_point，中心坐标列表center_xy_copy，子图ax,摄像头位置position
#计算拟合的列直线的斜率k_list和截距b_list
#将每列的拟合点坐标到总的列表points_for_col_regression中
#将每列的拟合矩形坐标对列表添加到总的列表coordinates_for_col_regression中
#无返回值
def loop_all_col_regression(first_point, first_point_flag, center_xy_copy, ax, position):
    #开始拟合列直线
    center_xy_copy = list(center_xy)  # 复制中心坐标列表
    coordinates_copy = list(coordinates)  # 复制剩余矩形坐标列表
    i = 1 # 用于计数
    # 循环拟合每列直线，直到所有中心坐标都被拟合完毕
    #最后得到列拟合点坐标列表 points_for_col_regression
    #得到列拟合矩形坐标列表 coordinates_for_col_regression
    while center_xy_copy:
        if i == 1:
            print("第", i, "次拟合：")
            if first_point_flag:
                points_for_regression ,coordinates_for_col_regression = list(loop_col_regression(first_point, center_xy_copy, ax))  # 拟合第一列直线，返回拟合点坐标列表和拟合矩形坐标列表
            else:
                points_for_regression ,coordinates_for_col_regression = single_first_col_regression(center_xy_copy,position)  # 拟合第一列直线，返回拟合点坐标列表和拟合矩形坐标列表
            print("第", i, "次拟合点坐标：", points_for_regression)
            #print("第", i, "次拟合矩形坐标：", coordinates_for_col_regression)
            #print("剩余待拟合点的坐标(未删除相交矩形的)：", coordinates_copy)
            #coordinates_copy_rect=remove_plots_from_points(coordinates_copy,coordinates_for_col_regression)
            #print("剩余待拟合矩形的坐标：", coordinates_copy_rect)
        else:
            print("第", i, "次拟合：")
            points_for_regression ,coordinates_for_col_regression= list(loop_col_regression_best(find_nth_largest_y_coordinate_vertex(center_xy_copy, 1,position), center_xy_copy, ax, position))  # 拟合第i列直线，返回拟合点坐标列表
            #print("第", i, "列拟合点坐标：", points_for_regression)
        #print("第", i, "列拟合直线的斜率:",k_list[i-1],"和截距：",b_list[i-1] )
        #判断拟合的第一列直线是否与剩余矩形框相交，如果有相交的，则将相交的矩形的中心坐标加入到本列的拟合点中
        #print("第", i, "列拟合点坐标：", points_for_regression)
        #print("第", i, "列拟合矩形坐标：", coordinates_for_col_regression)
        #print("剩余待拟合点的坐标(未删除相交矩形的)：", center_xy_copy)
        #print("剩余待拟合矩形的坐标(未删除相交矩形的)：", coordinates_copy)
        #print((points_for_regression))
        flag,center_point=check_line_intersects_rectangles(col_k_list[i-1], col_b_list[i-1],remove_plots_from_points(coordinates_copy,coordinates_for_col_regression) )
        if flag:
            print("第", i, "列拟合的直线与剩余相交矩形的中心坐标为：",center_point)
            #print("相交矩形的矩形框为：",center_to_rectangle[center_point[0]])
            #coordinates_copy = list(remove_plots_from_points(coordinates_copy,[center_to_rectangle[center_point[0]]])) #删除相交矩形的坐标
            if position==1:
                for point in center_point:
                    if point[0]>points_for_regression[0][0]:
                        points_for_regression.extend(center_point) #将相交矩形的中心坐标加入到本列的拟合点中
                        coordinates_for_col_regression.append(center_to_rectangle[point]) #将相交矩形的中心坐标对应的矩形坐标对加入到本列的拟合矩形坐标对中
                        center_xy_copy = list(remove_plots_from_points(center_xy_copy,center_point))  
                        coordinates_copy = list(remove_plots_from_points(coordinates_copy,coordinates_for_col_regression))
            elif position==-1:
                for point in center_point:
                    if point[0]<points_for_regression[0][0]:
                        points_for_regression.extend(center_point) #将相交矩形的中心坐标加入到本列的拟合点中
                        coordinates_for_col_regression.append(center_to_rectangle[point]) #将相交矩形的中心坐标对应的矩形坐标对加入到本列的拟合矩形坐标对中 
                        center_xy_copy = list(remove_plots_from_points(center_xy_copy,center_point))  
                        coordinates_copy = list(remove_plots_from_points(coordinates_copy,coordinates_for_col_regression))
            else:
                points_for_regression.extend(center_point) #将相交矩形的中心坐标加入到本列的拟合点中
                coordinates_for_col_regression.append(center_to_rectangle[point]) #将相交矩形的中心坐标对应的矩形坐标对加入到本列的拟合矩形坐标对中 
                center_xy_copy = list(remove_plots_from_points(center_xy_copy,center_point))  
                coordinates_copy = list(remove_plots_from_points(coordinates_copy,coordinates_for_col_regression))            
        #print("第", i, "次拟合点坐标：", points_for_regression)
            #print("剩余待拟合矩形的坐标(删除相交矩形的)：", coordinates_copy)
        #else:
            #print("第", i, "列拟合的直线与剩余矩形框不相交，无遗漏矩形")
        #print("第", i, "列拟合点坐标：", points_for_regression)
        else:
            print("第", i, "列拟合的直线与剩余矩形框不相交")
        #print("第", i, "次拟合点坐标：", points_for_regression)
        final_points_for_col_regression=sort_points_by_ymax_to_ymin(points_for_regression) #对拟合的点坐标按y值从大到小排序
        #print("排序后第", i, "列拟合点坐标：", final_points_for_col_regression)
        new_points_for_col_regression=remove_duplicates(final_points_for_col_regression) #删除重复的点坐标
        print("第", i, "次拟合点坐标：", new_points_for_col_regression)
        points_for_col_regression.append(new_points_for_col_regression)  # 将每列的拟合点坐标列表添加到总的列表中
        #print("第", i, "列拟合点坐标：", points_for_col_regression[i-1])
        center_xy_copy = list(remove_plots_from_points(center_xy_copy, points_for_regression))  # 从中心坐标列表中删除拟合过的坐标
        print("拟合第", i, "列后的剩余点坐标：", center_xy_copy)
        i+=1
        print('\n')




#定义对拟合的每一列进行检查的函数(主要是合并相交矩形)
#传入拟合好的每一列的拟合点坐标列表points_for_col_regression，子图ax，行数nums_of_row，摄像头位置position
#更新每一列的拟合点坐标列表points_for_col_regression
#返回更新后的center_xy
def all_col_regression_check(points_for_col_regression,ax,nums_of_row,position): 
    #对拟合好的每一列进行检查，看是否存在同列中心坐标对应的矩形有相交的情况，若有，则将该两相交矩形的中心坐标从该列的拟合点中删除，
    #在该列拟合点坐标列表中增加由两相交矩形合并而成的矩形中心坐标，并将该矩形的中心坐标加入到该列的拟合点中，并将该矩形的坐标对加入到该列的拟合矩形坐标对中
    for i in range(len(points_for_col_regression)):
        cols_points=points_for_col_regression[i]
        col_inter_flag=False
        end_flag=False
        count=0
        while len(points_for_col_regression[i])>nums_of_row and end_flag==False:  #拟合后每列坐标点个数大于行数才进行判断是否存在相交的情况
            #print("第", i+1, "列拟合点坐标个数不等于行数，进行相交矩形检查")
            #print("第", i+1, "列拟合点坐标：", cols_points)
            if count>=20: #若相交矩形检查次数大于10次，则退出循环
                end_flag=True
                break
            count+=1
            for j in range(len(cols_points)-1): #遍历该列的拟合点坐标，判断是否存在相交的情况
                point1=cols_points[j]  #第一个点
                point2=cols_points[j+1]  #第二个点
                f,rect,intersection_area_ratio=merge_rectangles(center_to_rectangle[point1],center_to_rectangle[point2],max_intersection_area_ratio) #判断两个矩形是否有重叠部分，并计算相交面积占比
                if f :#存在相交的情况
                    col_inter_flag=True #该列存在相交的情况
                    points_for_col_regression[i]=remove_plots_from_points(points_for_col_regression[i],[point1,point2]) #将相交的两个点从该列的拟合点坐标列表中删除
                    merged_center_point=((rect[0][0]+rect[1][0])/2,(rect[0][1]+rect[1][1])/2) #计算合并后的矩形中心坐标
                    points_for_col_regression[i].append(merged_center_point) #在该列的拟合点坐标列表中增加由两相交矩形合并而成的矩形中心坐标
                    points_for_col_regression[i]=sort_points_by_ymax_to_ymin(points_for_col_regression[i]) #对该列的拟合点坐标列表按y值从大到小排序
                    center_to_rectangle[merged_center_point]=rect #将合并后的矩形的坐标对加入到中心坐标to矩形坐标对的字典中
                    rectangle_to_center[rect]=merged_center_point  #将合并后的矩形的中心坐标加入到矩形坐标对to中心坐标的字典中
                    rect1=patches.Rectangle(rect[0],rect[1][0]-rect[0][0],rect[1][1]-rect[0][1],linewidth=2,edgecolor='green',facecolor='none') #绘制合并后的矩形
                    #ax.add_patch(rect1)
                    ax.text(rect[0][0],rect[1][1],str(round(intersection_area_ratio,4)),color='red',fontsize=10) #显示相交面积占比
                    if len(points_for_col_regression[i])==nums_of_row: #若该列的拟合点坐标列表中点个数等于行数，则退出循环
                        end_flag=True
                        break
                else:
                    if j==len(cols_points)-2: #若遍历到最后一个点，则退出循环
                        #end_flag=True
                        break
                    else:
                        continue
        if col_inter_flag==False and len(points_for_col_regression[i])>nums_of_row: #若该列的拟合点坐标列表中点个数大于行数，但不存在相交的情况，则对相交面积占比大于0.005的矩形进行合并
            rect_for_col_regression=[] #用于存储该列的拟合矩形坐标对
            for point in cols_points: #将该列的拟合点坐标列表转为矩形坐标对列表
                rect_for_col_regression.append(center_to_rectangle[point])
            fff,rect1,rect2,s=find_intersect_max_area_rectangles(rect_for_col_regression) #找出两相交矩形面积最大的两个矩形
            if fff and rect1!=None and rect2!=None : #若存在两相交矩形，则进行合并
                point1=rectangle_to_center[rect1]
                point2=rectangle_to_center[rect2]
                #print(point1)
                #print(point2)
                ff,rect,intersection_area_ratio=merge_rectangles(rect1,rect2,0.005) #判断两个矩形是否有重叠部分，并计算相交面积占比
                points_for_col_regression[i]=remove_plots_from_points(points_for_col_regression[i],[point1,point2]) #将相交的两个点从该列的拟合点坐标列表中删除
                merged_center_point=((rect[0][0]+rect[1][0])/2,(rect[0][1]+rect[1][1])/2) #计算合并后的矩形中心坐标
                points_for_col_regression[i].append(merged_center_point) #在该列的拟合点坐标列表中增加由两相交矩形合并而成的矩形中心坐标
                points_for_col_regression[i]=sort_points_by_ymax_to_ymin(points_for_col_regression[i]) #对该列的拟合点坐标列表按y值从大到小排序
                center_to_rectangle[merged_center_point]=rect #将合并后的矩形的坐标对加入到中心坐标to矩形坐标对的字典中
                rectangle_to_center[rect]=merged_center_point  #将合并后的矩形的中心坐标加入到矩形坐标对to中心坐标的字典中
                rect1=patches.Rectangle(rect[0],rect[1][0]-rect[0][0],rect[1][1]-rect[0][1],linewidth=2,edgecolor='green',facecolor='none') #绘制合并后的矩形
                #ax.add_patch(rect1)
                ax.text(rect[0][0],rect[1][1],str(round(intersection_area_ratio,4)),color='red',fontsize=10) #显示相交面积占比
                if len(points_for_col_regression[i])==nums_of_row: #若该列的拟合点坐标列表中点个数等于行数，则退出循环
                    break 

    #按照每一列的第一个坐标的x值列数进行排序,position==1时，从右侧开始，反之从左侧开始
    if position==1:
        points_for_col_regression=sorted(points_for_col_regression,key=lambda x:x[0][0],reverse=True)
    else:   
        points_for_col_regression=sorted(points_for_col_regression,key=lambda x:x[0][0])
    #因为进行了合并，所以每列拟合点的数量可能大于行数，所以需要更新拟合点坐标
    #更新中心坐标列表
    center_xy=[]
    for i in range(len(points_for_col_regression)):
        for j in range(len(points_for_col_regression[i])):
            center_xy.append(points_for_col_regression[i][j])
    return center_xy




#判断列直线拟合是否正确的函数
#传入拟合好的每一列的拟合点坐标列表points_for_col_regression，行数nums_of_row，摄像头位置position
#返回True或False，True表示所有列都拟合正确，False表示有列的拟合点数量不等于行数
def flag_col_regression(points_for_col_regression,nums_of_row,position):
    col_flag=True#判断是否所有列都拟合正常
    num_cols = len(points_for_col_regression)  # 列数
    per_col_lengths = [len(points) for points in points_for_col_regression]  # 每列拟合坐标的数量
    for i in range(len(points_for_col_regression)):
        if len(points_for_col_regression[i])!=nums_of_row: #判断每列拟合点的数量是否等于行数
            col_flag=False #若有列的拟合点数量不等于行数，则置col_flag为False

    #打印输出每列拟合的点坐标（正常应为从左到右1-5列）
    for i in range(len(points_for_col_regression)):
        if position==1:
            print("第",i+1,f"列 : 拟合点坐标为：", points_for_col_regression[num_cols-1-i])
        else:
            print("第",i+1,f"列 : 拟合点坐标为 ", points_for_col_regression[i])
    #打印输出每列拟合点的数量，正常应为1-5列
    if position==1:
        print("每列拟合坐标数量：", per_col_lengths[::-1])
    else:
        print("每列拟合坐标数量：", per_col_lengths)  
    return col_flag


##绘制中心点和矩形框
def draw_center_point_and_rect(center_xy,ax,img_size):
#绘制中心点和矩形框
    for point in center_xy:
        #绘制中心点
        ax.plot(point[0], point[1], 'o', color='blue', markersize=5)
        #绘制矩形框
        rect=center_to_rectangle[point]
        x1=rect[0][0]
        y1=rect[0][1]
        width=rect[1][0]-x1
        height=rect[1][1]-y1
        rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)


#定义列直线拟合正确，直接进行行直线拟合的函数
#传入拟合好的每一列的拟合点坐标列表points_for_col_regression，摄像头位置position
#得到每一行的拟合坐标points_for_row_regression
#计算每一行的拟合直线，得到每一行的斜率row_k_list和截距row_b_list
#构造坐标矩阵，Matrix[i][j]表示第i+1行第j+1列的中心坐标
#构造中心坐标-行列字典，points_to_matrix_dict[center_xy] = (i+1, j+1)，通过中心坐标获取所在的行与列
def row_regression_by_col_regression(points_for_col_regression,position):
    #拟合行直线
    row_num=len(points_for_col_regression[0])#获取行数
    col_num=len(points_for_col_regression) #获取列数
    # 遍历列获取每行的拟合点坐标列表
    #拟合每一行的直线
    for i in range(row_num):
        row_points=[]
        for j in range(col_num):
            row_points.append(points_for_col_regression[j][i])
        points_for_row_regression.append(row_points)
        k,b= loop_row_regression(row_points)#绘制每行的拟合直线
        row_k_list.append(k)
        row_b_list.append(b)
        #print("第", i+1, "行拟合直线的斜率:",k,"和截距：",b )
        row_points=sorted(row_points,key=lambda x:x[0])#按x值从小到大排序
        print("第", i+1, "行拟合点坐标：", row_points)
    # 构造坐标矩阵,Matrix[i][j]表示第i+1行第j+1列的中心坐标
    Matrix = construct_coordinate_matrix(points_for_row_regression, points_for_col_regression,position)  
    #print("坐标矩阵：", Matrix)
    # 构造中心坐标-行列字典,points_to_matrix_dict[center_xy] = (i+1, j+1),通过中心坐标获取所在的行与列
    points_to_matrix_dict = create_coordinate_dictionary(Matrix)
    #print("中心坐标-行列字典：", points_to_matrix_dict)
    return Matrix,points_to_matrix_dict


#定义绘制拟合列直线的函数
def draw_col_line(col_k_list,col_b_list,points_for_col_regression,ax,img_size):
    #绘制拟合列直线
    for i in range(len(points_for_col_regression)):
        draw_line(col_k_list[i],col_b_list[i],ax,'green',0,img_size[0]) #绘制拟合的直线
        #plt.text(points_for_col_regression[i][0][0], points_for_col_regression[i][0][1], str(i+1), color='green', fontsize=20)  # 绘制拟合点的序号


# 定义绘制拟合行直线的函数
def draw_row_line(row_k_list,row_b_list,points_for_row_regression,ax,img_size):
    for i in range(len(points_for_row_regression)):
        draw_line(row_k_list[i],row_b_list[i],ax,'orange',0,img_size[0]) #绘制拟合的直线

#定义显示中心坐标的行与列的函数
def display_row_col(points_to_matrix_dict,center_xy,ax):
    for point in center_xy:
        ax.text(point[0], point[1], str(points_to_matrix_dict[point]), color='black', fontsize=10)  # 绘制中心坐标对应的行列号


#定义设置坐标轴范围和保存路径等的函数
def set_axis_and_save(col_flag,ax,img_name,img_size,save_dir):
    # 设置图形的标题和坐标轴标签
    ax.set_title(img_name)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # 设置坐标轴范围
    ax.set_xlim(0, img_size[0])
    ax.set_ylim(img_size[1], 0)  # 颠倒y轴坐标
    if col_flag:
        #保存图形
        save_name = img_name + ".jpg"  # 使用 img_name 作为文件名，并添加文件扩展名
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path)
        timer = threading.Timer(0.05, close_plot)
        timer.start()
        #进入Tkinter的主事件循环
        plt.get_current_fig_manager().window.mainloop()
        # 显示图形
    else:#列直线拟合错误，只进行列直线拟合
        #保存图形
        save_name = "列拟合错误—"+img_name + ".jpg"  # 使用 列拟合错误—img_name 作为文件名，并添加文件扩展名
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path)
        timer = threading.Timer(0.05, close_plot)
        timer.start()
        #进入Tkinter的主事件循环
        plt.get_current_fig_manager().window.mainloop()
        # 显示图形
    #plt.grid(True)
    #plt.show()


#定义计算中心坐标的函数
def calculate_center_xy(coordinates):
    # 遍历坐标对，计算中心坐标
    for coord_pair in coordinates:
        # 提取左上角和右下角坐标
        (x1, y1), (x2, y2) = coord_pair
        # 计算矩形的宽度和高度
        #width = x2 - x1
        #height = y2 - y1
        # 计算矩形的中心坐标
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        center_xy.append((center_x,center_y)) # 存储中心坐标
        center_to_rectangle[(center_x,center_y)]=coord_pair # 存储中心坐标与矩形坐标对的对应关系
        rectangle_to_center[coord_pair]=(center_x,center_y) # 存储矩形坐标对与中心坐标的对应关系
        # 创建矩形对象并添加到子图中
        #rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='r', facecolor='none')
        #ax.add_patch(rect)
        # 绘制中心点
        #ax.plot(center_x, center_y, marker='o', markersize=5, color='blue')
    center_xy_array = np.array(center_xy)#列表转为numpy数组 
    return center_xy,center_xy_array,len(center_xy) #返回中心坐标列表，中心坐标numpy数组，中心坐标个数


#当缺失第一个坐标时，对第一列进行单独拟合
def single_first_col_regression(center_xy,position):
    rects=[]
    first_col_points=[]
    first_col_rects=[]
    for i in range(len(center_xy)):
        rects.append(center_to_rectangle[center_xy[i]])
    if position==-1: #摄像头在左侧
        rects=sorted(rects,key=lambda x:x[0][0])#按矩形坐标对的左上角x值从小到大排序
        for i in range(len(rects)):
            if i<=4:
                first_col_points.append(rectangle_to_center[rects[i]])
        k,b=linear_regression(np.array(first_col_points))
    if position==1: #摄像头在右侧
        rects=sorted(rects,key=lambda x:x[1][0],reverse=True)#按矩形坐标对的右下角x值从大到小排序
        for i in range(len(rects)):
            if i<=4:
                first_col_points.append(rectangle_to_center[rects[i]])
        k,b=linear_regression(np.array(first_col_points))
    col_k_list.append(k)
    col_b_list.append(b)
    for i in range(len(first_col_points)):
        first_col_rects.append(center_to_rectangle[first_col_points[i]])
    return first_col_points,first_col_rects


#当缺失最后一个坐标时，对第一行进行单独拟合
def single_first_row_regression(points_for_col_regression):
    first_row_points=[]
    for i in range(len(points_for_col_regression)):
        if i!=0:
            first_row_points.append(points_for_col_regression[i][0])
    k,b=linear_regression(np.array(first_row_points))
    #row_k_list.append(k)
    #row_b_list.append(b)
    return k,b,first_row_points


#计算两条直线的交点
def intersection(k1,b1,k2,b2):
    x=(b2-b1)/(k1-k2)
    y=k1*x+b1
    return (x,y)


#获取一组中心坐标对应矩形的平均宽度和高度
def get_width_and_height(points):
    width=0
    height=0
    for point in points:
        rect=center_to_rectangle[point]
        (x1, y1), (x2, y2) = rect
        width = width + x2 - x1
        height = height + y2 - y1
    width=width/len(points)
    height=height/len(points)
    return  width,height


#只有第一个点缺失，通过第一列和第一行的交点，人为补出
def fill_missing_first_point(first_point,first_point_flag,center_xy,position):
        print("第一个点不正确，对第一列单独拟合")
        loop_all_col_regression(first_point,first_point_flag,center_xy,ax,position)#进行列直线拟合
        #print("单独拟合的第一列拟合点坐标：",points_for_col_regression[0])
        #print("单独拟合的第一列拟合直线的斜率:",col_k_list[0],"和截距：",col_b_list[0])
        draw_col_line(col_k_list,col_b_list,points_for_col_regression,ax,img_size)#绘制拟合列直线
        for i in range(len(points_for_col_regression)):
            print("第",i+1,f"列 : 拟合点坐标为 ", points_for_col_regression[i])
        #print("拟合第一行")
        row_k1,row_b1,first_row_points=single_first_row_regression(points_for_col_regression)
        #print("单独拟合的第一行拟合点坐标：",first_row_points)
        #print("单独拟合的第一行拟合直线的斜率:",row_k1,"和截距：",row_b1)
        first_point=(intersection(row_k1,row_b1,col_k_list[0],col_b_list[0]))
        first_point=(int(first_point[0]),int(first_point[1]))
        #print("交点坐标为：",first_point)
        #plt.plot(first_point[0],first_point[1],'o',color='red',markersize=5)
        points_for_col_regression[0].append(first_point)
        points_for_col_regression[0]=sorted(points_for_col_regression[0],key=lambda y:y[1],reverse=True)
        center_xy.append(first_point)
        center_xy_array=np.array(center_xy)
        p00=points_for_col_regression[0][0]
        p01=points_for_col_regression[0][1]
        p10=points_for_col_regression[1][0]
        p11=points_for_col_regression[1][1]
        p123=[p01,p10,p11,]
        width,height=get_width_and_height(p123)
        width=int(width)
        height=int(height)
        #print("修正后的矩形的宽度和高度：",width,height)
        first_rect=((p00[0]-width/2,p00[1]-height/2),(p00[0]+width/2,p00[1]+height/2))
        #print("修正后的第一个矩形坐标对：",first_rect)
        center_to_rectangle[first_point]=first_rect
        rectangle_to_center[first_rect]=first_point

#绘制列直线，行直线，显示行列序号
def display_colline_rowline_and_num():
    draw_col_line(col_k_list,col_b_list,points_for_col_regression,ax,img_size)#绘制拟合列直线
    draw_row_line(row_k_list,row_b_list,points_for_row_regression,ax,img_size)#绘制拟合行直线
    display_row_col(points_to_matrix_dict,center_xy,ax)#绘制中心坐标对应的行列号


#对center_xy对应矩形按需要进行排序（x坐标从小到大或者x坐标从大到小），返回排序后的列表
def sort_center_xy_by_rect_x_(center_xy,f):
    coordinates=[]
    sorted_center_xy=[]
    for i in range(len(center_xy)):
        coordinates.append(center_to_rectangle[center_xy[i]])
    if f:#按x坐标从大到小排序
        coordinates=sorted(coordinates,key=lambda x:x[1][0],reverse=True)
    else:#按x坐标从小到大排序
        coordinates=sorted(coordinates,key=lambda x:x[0][0])
    for i in range(len(coordinates)):
        sorted_center_xy.append(rectangle_to_center[coordinates[i]])
    return sorted_center_xy



#计算center_xy对应的矩形框，x最小值的5个矩形左上角x坐标的方差d1，x最大值的5个矩形右下角x坐标的方差d2
def calculate_variance(center_xy):
    d1=0 #右边五个矩形的x坐标的方差
    d2=0 #左边五个矩形的x坐标的方差
    #先计算右边五个矩形的x坐标的方差
    rects=[]
    rects_x=[]
    sorted_center_xy=sort_center_xy_by_rect_x_(center_xy,True)
    for i in range(4):
        rects.append(center_to_rectangle[sorted_center_xy[i]])
        rects_x.append(rects[i][1][0])
    d1=np.var(rects_x)
    #再计算左边五个矩形的x坐标的方差
    rects=[]
    rects_x=[]
    sorted_center_xy=sort_center_xy_by_rect_x_(center_xy,False)
    for i in range(4):
        rects.append(center_to_rectangle[sorted_center_xy[i]])
        rects_x.append(rects[i][0][0])
    d2=np.var(rects_x)
    return d1,d2


#进一步判断摄像头位置
def judge_position(points,img_size):
    flag, first_point = judge_camera_position_best(points,img_size)
    d1,d2=calculate_variance(center_xy)
    print("d1=",d1,"d2=",d2)
    f=0
    
    if d1>d2:
        #print("摄像头在左侧")
        f=-1
    else:
        #print("摄像头在右侧")
        f=1
    ff=flag*0.5+f*0.6
    if ff>0:
        print("摄像头在右侧")
        return 1,first_point
    else:
        print("摄像头在左侧")
        return -1,first_point




#####################################################进行行列直线拟合##############################################################
#计算中心坐标
center_xy,center_xy_array,nums=calculate_center_xy(coordinates) 
#draw_center_point_and_rect(center_xy,ax,img_size) #绘制中心点和矩形框
#判断摄像头位置并获取用于拟合的第一个中心坐标
position,first_point=judge_position(center_xy,img_size)#判断摄像头位置,获得第一个中心坐标 
print("第一个点坐标为：",first_point)#打印输出用于拟合的第一个中心坐标
#print("中心坐标为：",center_xy)#打印输出中心坐标
#print("中心坐标个数为：",len(center_xy))

print("中心坐标个数为：",nums)
first_point_flag=is_first_point_correct(first_point, center_xy, position)#第一个点是否正确的标志位
col_flag=False#每列直线拟合结果是否正确的标志位
print("第一个点是否正确的标志位：",first_point_flag)
if (first_point_flag==0 and nums<29 )or (nums>36 and first_point_flag==0)  or (nums<29 and first_point_flag==1) :#中心坐标个数小于29, 第一个点不正确，有遗漏矩形也有误检测(相交矩形)
    if first_point_flag==0:
        fill_missing_first_point(first_point,first_point_flag,center_xy,position)#对第一个点进行单独拟合
        center_xy= all_col_regression_check(points_for_col_regression,ax,nums_of_row,position)#检查每列直线拟合结果，主要是合并相交矩形
        draw_col_line(col_k_list,col_b_list,points_for_col_regression,ax,img_size)#绘制拟合列直线
    else:
        loop_all_col_regression(first_point,first_point_flag,center_xy,ax,position)#进行列直线拟合
        center_xy= all_col_regression_check(points_for_col_regression,ax,nums_of_row,position)#检查每列直线拟合结果，主要是合并相交矩形
        col_flag=flag_col_regression(points_for_col_regression,nums_of_row,position)#检查每列直线拟合结果是否正确
        if col_flag: #每列直线拟合结果正确，进行行直线拟合
            print("每列直线拟合结果正确，进行行直线拟合")
            Matrix,points_to_matrix_dict=row_regression_by_col_regression(points_for_col_regression,position)#进行行直线拟合
            display_colline_rowline_and_num() #绘制列直线，行直线，显示行列序号
            #draw_center_point_and_rect(center_xy,ax,img_size) #绘制中心点和矩形框
            #set_axis_and_save(col_flag,ax,img_name,img_size,save_dir)#设置坐标轴范围和保存路径等
        else:#每列直线拟合结果错误，暂时只进行列直线拟合
            draw_col_line(col_k_list,col_b_list,points_for_col_regression,ax,img_size)#绘制拟合列直线
            #draw_center_point_and_rect(center_xy,ax,img_size) #绘制中心点和矩形框
            #set_axis_and_save(col_flag,ax,img_name,img_size,save_dir)#设置坐标轴范围和保存路径等
    draw_center_point_and_rect(center_xy,ax,img_size) #绘制中心点和矩形框
    set_axis_and_save(col_flag,ax,img_name,img_size,save_dir)#设置坐标轴范围和保存路径等
    exit()#退出程序

#first_point_flag=1#暂时将第一个点标志位设为1，后续需要修改
if nums>=30 and first_point_flag: #中心坐标个数大于30, 第一个点正确,没有遗漏矩形，可以尝试进行拟合
    print("中心坐标个数大于30, 第一个点正确,没有遗漏矩形，可以尝试进行拟合")
    loop_all_col_regression(first_point,first_point_flag,center_xy,ax,position)#进行列直线拟合
    center_xy= all_col_regression_check(points_for_col_regression,ax,nums_of_row,position)#检查每列直线拟合结果，主要是合并相交矩形
    col_flag=flag_col_regression(points_for_col_regression,nums_of_row,position)#检查每列直线拟合结果是否正确
    if col_flag: #每列直线拟合结果正确，进行行直线拟合
        print("每列直线拟合结果正确，进行行直线拟合")
        Matrix,points_to_matrix_dict=row_regression_by_col_regression(points_for_col_regression,position)#进行行直线拟合
        display_colline_rowline_and_num() #绘制列直线，行直线，显示行列序号
        draw_center_point_and_rect(center_xy,ax,img_size) #绘制中心点和矩形框
        set_axis_and_save(col_flag,ax,img_name,img_size,save_dir)#设置坐标轴范围和保存路径等
    else:#每列直线拟合结果错误，暂时只进行列直线拟合
        draw_col_line(col_k_list,col_b_list,points_for_col_regression,ax,img_size)#绘制拟合列直线
        draw_center_point_and_rect(center_xy,ax,img_size) #绘制中心点和矩形框
        set_axis_and_save(col_flag,ax,img_name,img_size,save_dir)#设置坐标轴范围和保存路径等

if nums<30  : #中心坐标个数小于30, 有遗漏矩形
    print("中心坐标个数小于30, 有遗漏矩形")
    if first_point_flag!=1: #第一个点不正确，对第一列单独拟合
        fill_missing_first_point(first_point,first_point_flag,center_xy,position)#对第一个点进行单独拟合
        center_xy= all_col_regression_check(points_for_col_regression,ax,nums_of_row,position)#检查每列直线拟合结果，主要是合并相交矩形
        Matrix,points_to_matrix_dict=row_regression_by_col_regression(points_for_col_regression,position)#进行行直线拟合
        display_colline_rowline_and_num() #绘制列直线，行直线，显示行列序号
        col_flag=True
        draw_center_point_and_rect(center_xy,ax,img_size) #绘制中心点和矩形框
        set_axis_and_save(col_flag,ax,img_name,img_size,save_dir)#设置坐标轴范围和保存路径等
    else: 
        first_point_flag=0#将第一个点标志位设为0，后续需要修改
        fill_missing_first_point(first_point,first_point_flag,center_xy,position)#对第一个点进行单独拟合
        #loop_all_col_regression(first_point,first_point_flag,center_xy,ax,position)#进行列直线拟合
        center_xy= all_col_regression_check(points_for_col_regression,ax,nums_of_row,position)#检查每列直线拟合结果，主要是合并相交矩形
        col_flag=flag_col_regression(points_for_col_regression,nums_of_row,position)#检查每列直线拟合结果是否正确
        if col_flag: #每列直线拟合结果正确，进行行直线拟合
            Matrix,points_to_matrix_dict=row_regression_by_col_regression(points_for_col_regression,position)#进行行直线拟合
            display_colline_rowline_and_num() #绘制列直线，行直线，显示行列序号
        draw_col_line(col_k_list,col_b_list,points_for_col_regression,ax,img_size) 
        draw_center_point_and_rect(center_xy,ax,img_size) #绘制中心点和矩形框
        set_axis_and_save(col_flag,ax,img_name,img_size,save_dir)#设置坐标轴范围和保存路径等


if nums>=30 and first_point_flag!=1 : #中心坐标个数大于30, 第一个点不正确，有遗漏矩形也有误检测(相交矩形)
    print("中心坐标个数大于等于30, 但是第一个点不正确，可能有遗漏矩形也有误检测（相交矩形）")
    fill_missing_first_point(first_point,first_point_flag,center_xy,position)#对第一个点进行单独拟合
    center_xy= all_col_regression_check(points_for_col_regression,ax,nums_of_row,position)#检查每列直线拟合结果，主要是合并相交矩形
    Matrix,points_to_matrix_dict=row_regression_by_col_regression(points_for_col_regression,position)#进行行直线拟合
    display_colline_rowline_and_num() #绘制列直线，行直线，显示行列序号
    col_flag=True
    draw_center_point_and_rect(center_xy,ax,img_size) #绘制中心点和矩形框
    set_axis_and_save(col_flag,ax,img_name,img_size,save_dir)#设置坐标轴范围和保存路径等

print("每列直线拟合结果是否正确：",col_flag)
