# Row-and-column-coordinates-fit
Based on the data of desk prediction frame in the classroom detected by yolov5, such as desk length and width and center point coordinates, an algorithm was designed to judge the rows and columns of desks, and finally the fitted row and row coordinates of each desk were obtained and displayed with images. 

1.Clone the Repository  

```
git clone https://github.com/HuiFaLiu/Row-and-column-coordinates-fit.git
```
##2.Data preparation   
Download and decompress the Prediction_frame_data.zip file. 

3.Create a Python virtual environment
```
conda create -n test1 python=3.9.19
conda activate test1
```

4.Install the required packages
```
pip install -r requirements.txt
```

5.Test a single image   
open the test.py file    
motify the all_img_flag to False    
//False indicates that only a single image is fitted, and True indicates that all images in the entire folder are fitted
motify the img_path to the path of the image you want to test   
such as:
```
img_path = 'C:/Users/27210/Desktop/课桌行列/test-data3/labels/vacant_1370.txt' //txt file located in labels in the Prediction_frame_data.zip file
```
motify save_dir to the path where you want to save the result image   
such as: 
```
save_dir = 'C:/Users/27210/Desktop/result'
```
run the test.py file
```
python test.py
```
The result will be displayed in the console and see the fitted row and row coordinates of the desk in the image.


6.Test all images in the folder   
open the test.py file    
motify the all_img_flag to True    
//False indicates that only a single image is fitted, and True indicates that all images in the entire folder are fitted
motify the img_path to the path of the folder where the images are located   
such as:
```
img_path = 'C:/Users/27210/Desktop/课桌行列/test-data3/labels' //folder located in labels in the Prediction_frame_data.zip file
motify save_dir to the path where you want to save the result images   
such as: 
```
save_dir = 'C:/Users/27210/Desktop/result'
```
Appends the information printed by the terminal to the specified file   
such as:
```
redirect_output_to_files("C:/Users/27210/Desktop/test_8/normal_output.txt", "c:/Users/27210/Desktop/test_8/error_output.txt")  //first file path is the normal output file path, and the second file path is the error output file path
```
open the test_all_image.py file    
motify txt_dir to the path of the folder where the txt files are located   
such as:
```
txt_dir = "C:/Users/27210/Desktop/test-data3/labels"   //folder located in labels in the Prediction_frame_data.zip file
```
run the test_all_image.py file
```
python test_all_image.py
```
The result will be displayed in the console and see the fitted row and row coordinates of the desk in the image. The result images will be saved in the specified folder.
