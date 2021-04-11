import os
import cv2
import time

img_root = "brackish/"
path = "./"
filelist = os.listdir(img_root)
fps = 1
# file_path='saveVideo.avi' # 导出路径MJPG
# file_path='saveVideo'+str(int(time.time()))+'.mp4' # 导出路径DIVX/mp4v
file_path = '{}brackish.mp4'.format(path)  # 导出路径DIVX/mp4v
# print('F:/Work/Supermarket/Activities/Demo/gty_cz/rect_00000001_1.jpg')
img = cv2.imread(img_root + '1.jpg')
imgInfo = img.shape
size = (imgInfo[1], imgInfo[0])

#可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg: sudo apt-get install ffmepg
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # avi
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

videoWriter = cv2.VideoWriter(file_path, fourcc, fps, size)

# 这种情况更适合于照片是从"1.jpg" 开始，然后每张图片名字＋1的那种
# for i in range(8):
#     frame = cv2.imread(img_root+str(i+1)+'.jpg')
#     videoWriter.write(frame)

for item in filelist:
    if item.endswith('.jpg'):   #判断图片后缀是否是.jpg
        item = img_root + item
        img = cv2.imread(item) #使用opencv读取图像，直接返回numpy.ndarray 对象，通道顺序为BGR ，注意是BGR，通道值默认范围0-255。
        print(type(img))  # numpy.ndarray类型
        videoWriter.write(img)        #把图片写进视频

videoWriter.release() #释放