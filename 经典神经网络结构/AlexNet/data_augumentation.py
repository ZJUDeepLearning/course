from PIL import Image, ImageEnhance, ImageOps, ImageFile  
import numpy as np  
import random  
import threading, os, time  
import sys
import logging  
import cv2  
logger = logging.getLogger(__name__)  
ImageFile.LOAD_TRUNCATED_IMAGES = True  
  

class DataAugmentation:  
    """ 
    包含数据增强的八种方式 
    """  
  
  
    def __init__(self):  
        pass  
 
    @staticmethod  
    def openImage(image):  
        return Image.open(image, mode="r")  
 
    @staticmethod  
    def randomRotation(image, mode=Image.BICUBIC):  
        """ 
         对图像进行随机任意角度(0~360度)旋转 
        :param mode 邻近插值,双线性插值,双三次B样条插值(default) 
        :param image PIL的图像image 
        :return: 旋转转之后的图像 
        """  
        random_angle = np.random.randint(1, 360)  
        return image.rotate(random_angle, mode)  
        #return image.rotate(180, mode)  
  
    @staticmethod  
    def randomFlip(image):  
        #图像翻转（类似于镜像，镜子中的自己）  
        #FLIP_LEFT_RIGHT,左右翻转  
        #FLIP_TOP_BOTTOM,上下翻转  
        #ROTATE_90, ROTATE_180, or ROTATE_270.按照角度进行旋转，与randomRotate()功能类似  
        return image.transpose(Image.FLIP_LEFT_RIGHT)  
    
    @staticmethod  
    def Tranform(image):  
        #t图像变换  
        #im.transform(size, method, data) ⇒ image  
  
        #im.transform(size, method, data, filter) ⇒ image  
        #1：image.transform((300,300), Image.EXTENT, (0, 0, 300, 300))   
        #   变量data为指定输入图像中两个坐标点的4元组(x0,y0,x1,y1)。  
        #   输出图像为这两个坐标点之间像素的采样结果。  
        #   例如，如果输入图像的(x0,y0)为输出图像的（0，0）点，(x1,y1)则与变量size一样。  
        #   这个方法可以用于在当前图像中裁剪，放大，缩小或者镜像一个任意的长方形。  
        #   它比方法crop()稍慢，但是与resize操作一样快。  
        #2：image.transform((300,300), Image.AFFINE, (1, 2,3, 2, 1,4))  
        #   变量data是一个6元组(a,b,c,d,e,f)，包含一个仿射变换矩阵的第一个两行。  
        #   输出图像中的每一个像素（x，y），新值由输入图像的位置（ax+by+c, dx+ey+f）的像素产生，  
        #   使用最接近的像素进行近似。这个方法用于原始图像的缩放、转换、旋转和裁剪。  
        #3: image.transform((300,300), Image.QUAD, (0,0,0,500,600,500,600,0))  
        #   变量data是一个8元组(x0,y0,x1,y1,x2,y2,x3,y3)，它包括源四边形的左上，左下，右下和右上四个角。  
        #4: image.transform((300,300), Image.MESH, ())  
        #   与QUAD类似，但是变量data是目标长方形和对应源四边形的list。  
        #5: image.transform((300,300), Image.PERSPECTIVE, (1,2,3,2,1,6,1,2))  
        #   变量data是一个8元组(a,b,c,d,e,f,g,h)，包括一个透视变换的系数。  
        #   对于输出图像中的每个像素点，新的值来自于输入图像的位置的(a x + b y + c)/(g x + h y + 1),  
        #   (d x+ e y + f)/(g x + h y + 1)像素，使用最接近的像素进行近似。  
        #   这个方法用于原始图像的2D透视。  
        return image.transform((300,300), Image.EXTENT, (0, 0, 300, 300))  
 
    @staticmethod  
    def randomCrop(image):  
        """ 
        对图像随意剪切,考虑到图像大小范围(68,68),使用一个一个大于(36*36)的窗口进行截图 
        :param image: PIL的图像image 
        :return: 剪切之后的图像 
 
        """  
        image_width = image.size[0]  
        image_height = image.size[1]  
        crop_win_size = np.random.randint(40, 68)  
        random_region = (  
            (image_width - crop_win_size) >> 1, (image_height - crop_win_size) >> 1, (image_width + crop_win_size) >> 1,  
            (image_height + crop_win_size) >> 1)  
        return image.crop(random_region)  
 
    @staticmethod  
    def randomColor(image):  
        """ 
        对图像进行颜色抖动 
        :param image: PIL的图像image 
        :return: 有颜色色差的图像image 
        """  
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子  
        color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度  
        random_factor = np.random.randint(10, 21) / 10.  # 随机因子  
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度  
        random_factor = np.random.randint(10, 21) / 10.  # 随机因1子  
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度  
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子  
        return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度  
 
    @staticmethod  
    def randomGaussian(image, mean=0.2, sigma=0.3):  
        """ 
         对图像进行高斯噪声处理 
        :param image: 
        :return: 
        """  
  
        def gaussianNoisy(im, mean=0.2, sigma=0.3):  
            """ 
            对图像做高斯噪音处理 
            :param im: 单通道图像 
            :param mean: 偏移量 
            :param sigma: 标准差 
            :return: 
            """  
            for _i in range(len(im)):  
                im[_i] += random.gauss(mean, sigma)  
            return im  
  
        # 将图像转化成数组  
        img = np.asarray(image)  
        img.flags.writeable = True  # 将数组改为读写模式  
        width, height = img.shape[:2]  
        img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)  
        img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)  
        img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)  
        img[:, :, 0] = img_r.reshape([width, height])  
        img[:, :, 1] = img_g.reshape([width, height])  
        img[:, :, 2] = img_b.reshape([width, height])  
        return Image.fromarray(np.uint8(img))  
 
    @staticmethod  
    def saveImage(image, path):  
        image.save(path)  
  
  
def makeDir(path):  
    try:  
        if not os.path.exists(path):  
            if not os.path.isfile(path):  
                # os.mkdir(path)  
                os.makedirs(path)  
            return 0  
        else:  
            return 1  
    except Exception:  
        print (str(e)) 
        return -2  
  
  
def imageOps(func_name, image, des_path, file_name, times=5):  
    #funcMap = {"randomRotation": DataAugmentation.randomRotation,  
    #           "randomCrop": DataAugmentation.randomCrop,  
    #           "randomColor": DataAugmentation.randomColor,  
    #           "randomGaussian": DataAugmentation.randomGaussian  
    #           "randomFlip":DataAugmentation.randomFlip,  
    #           }  
    funcMap = {  
               "Tranform":DataAugmentation.Tranform  
               }  
    if funcMap.get(func_name) is None:  
        logger.error("%s is not exist", func_name)  
        return -1  
  
    for _i in range(0, times, 1):  
        new_image = funcMap[func_name](image)  
        DataAugmentation.saveImage(new_image, os.path.join(des_path, func_name + str(_i) + file_name))  
  
  
#opsList = {"randomFlip","randomRotation", "randomCrop", "randomColor", "randomGaussian"}  
opsList = { "randomFlip"}  
  
def threadOPS(path, new_path):  
    """ 
    多线程处理事务 
    :param src_path: 资源文件 
    :param des_path: 目的地文件 
    :return: 
    """  
    if os.path.isdir(path):  
        img_names = os.listdir(path)  
    else:  
        img_names = [path]  
    for img_name in img_names:  
        print (img_name)  
        tmp_img_name = os.path.join(path, img_name)  
        if os.path.isdir(tmp_img_name):  
            if makeDir(os.path.join(new_path, img_name)) != -1:  
                threadOPS(tmp_img_name, os.path.join(new_path, img_name))  
            else:  
                print ('create new dir failure')  
                return -1  
                # os.removedirs(tmp_img_name)  
        elif tmp_img_name.split('.')[1] != "DS_Store":  
            # 读取文件并进行操作  
            image = DataAugmentation.openImage(tmp_img_name)  
            threadImage = [0] * 5  
            _index = 0  
            for ops_name in opsList:  
                threadImage[_index] = threading.Thread(target=imageOps,  
                                                       args=(ops_name, image, new_path, img_name,))  
                threadImage[_index].start()  
                _index += 1  
                time.sleep(0.2)  
  
  
if __name__ == '__main__':  
    threadOPS(sys.argv[1],sys.argv[2])