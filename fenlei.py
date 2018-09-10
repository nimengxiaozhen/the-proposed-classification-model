# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 10:46:09 2018

@author: Administrator
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#设置默认显示参数
plt.rcParams['figure.figsize']=(10,10)     #图像显示大小
plt.rcParams['image.interpolation']='nearest'  #最邻近差值：像素为正方形
plt.rcParams['image.cmap']='gray'          #使用灰度输出而不是彩色输出

#加载caffe
caffe_root='C://Program Files (x86)//caffe-master//'
import caffe

#加载自己的分类模型
import os
if os.path.isfile(caffe_root+'examples//mnist//weitiao//5000-100//0.001_iter_5000.caffemodel'):
    print'Model found.'
    
    
#加载网络并设置输入预处理
caffe.set_mode_cpu()
model_def=caffe_root+'examples//mnist//weitiao//lenet.prototxt'  #模型的定义
model_weights=caffe_root+'examples//mnist//weitiao//5000-100//0.001_iter_5000.caffemodel'  #权重文件
net=caffe.Net(model_def,model_weights,caffe.TEST)


#设置输入预处理
#模型默认的输入图像格式为BGR，像素值的取值范围为【0,255】，同时每个像素值都减去了图像的平均
#值。Matplotlib加载matplotlib加载的图像的像素值位于[0,1]之间，并且格式是RGB格式，所以我们需要做一些变换。

#加载RS数据集的图像均值
mu=np.load(caffe_root+'examples//mnist//weitiao//mean.npy')
mu=mu.mean(1).mean(1)  #对所有像素值取平均以此获取BGR的均值像素值 
print'mean-substracted values:',zip('BGR',mu)

#对输入数据进行变换
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  
transformer.set_transpose('data', (2,0,1))      
#transformer.set_mean('data', mu)              #对于每个通道，都减去BGR的均值像素值  
#transformer.set_raw_scale('data', 255)        #将像素值从[0,255]变换到[0,1]之间  
#transformer.set_channel_swap('data', (2,1,0)) #交换通道，从RGB变换到BGR 

#cpu分类
# 设置输入图像大小  
net.blobs['data'].reshape(100,        # batch 大小  
                          1,         # 1-channelimages  
                          28, 28)  # 图像大小为:28x28 

#遍历文件夹下的图片
dir='H://zhuzhou//quxian//samples//erzhihua//' 
filelist=[]

filenames=os.listdir(dir)  #返回指定目录下的所有文件和文件名
filenames.sort(key= lambda x:int(x[:-4]))
print(filenames)

for fn in filenames:
    fullfilename=os.path.join(dir,fn)  #os.path.join---拼接路径
    filelist.append(fullfilename)
print (filelist)
    
#分类过程    
myarray=np.zeros(264196)
for i in range(0,len(filelist)):
    img=filelist[i]  #获取当前图片的路径
    print filenames[i]  #打印当前图片的名称
    im=caffe.io.load_image(img,False) #加载图片
    transformed_image = transformer.preprocess('data', im)  

    net.blobs['data'].data[...]=transformed_image
    output=net.forward()
    output_prob=output['prob'][0] 
    a=output_prob.argmax()#batch中第一张图像的概率值     
    myarray[i]=a
    print 'predicted class is:',a
 
print(myarray)
b=np.array(myarray).reshape(514,514)
print(b)

img=Image.fromarray(b)
img.save('H://zhuzhou//quxian//fenlei.tif')
img.show()