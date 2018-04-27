# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 20:52:13 2018

@author: yupeifeng

"""

import tensorflow as tf
import numpy as np

class AlexNet(object):
    """
        实现AlexNet
    """
    def __init__(self,x,keep_prob,num_classes,skip_layer,weights_path="DEFAULT"):
        """
            创建AlexNet的图
            参数：
                x：Input Tensor的Placeholder
                keep_prob：Drop的概率
                num_classes：类别数
                skip_layer：不同层的名字
                weight_path:权重路径
        """
        self.X=x
        self.NUM_CLASSES=num_classes
        self.KEEP_PROB=keep_prob
        self.SKIP_LAYER=skip_layer
        
        if weights_path=="DEFAULT":
            self.WEIGHTS_PATH='bvlc_alexnet.npy'
        else:
            self.WEIGHTS_PATH=weights_path
            
        self.create()
    
    def create(self):
        """
            创建计算图
        
            构建第一层：Conv(w ReLU) -> LRN -> Pool
        """
        
        conv1=conv(self.X,11,11,96,4,4,padding='VALID',name='conv1')
        norm1=lrn(conv1,2,1e-05,0.75,name='norm1')
        pool1=max_pool(norm1,3,3,2,2,padding='VALID',name='pool1')
        
        """
            构建第二层：Conv(w ReLU) -> LRN -> Pool with 2 groups
        """
        conv2 = conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        norm2 = lrn(conv2, 2, 1e-05, 0.75, name='norm2')
        pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')
        
        """
            构建第三层：Conv(w ReLU)
        """
        conv3=conv(pool2,3,3,384,1,1,name='conv3')
        
        """
            构建第四层：Conv(w ReLU) 分割到2组
        """
        conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')
        
        """
            构建第五层：Conv(w ReLU) -> Pool 分割到两个组
        """
        
        conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')
        
        """
            构建第六层：Flatten -> FC(w ReLU) -> Dropout
        """
        flattened=tf.reshape(pool5,[-1,6*6*256])
        fc6=fc(flattened,6*6*256,4096,name='fc6')
        dropout6=dropout(fc6,self.KEEP_PROB)
        
        """
            构建第7层：FC(w,ReLU) -> Dropout
        """
        fc7=fc(dropout6,4096,4096,name='fc7')
        dropout7=dropout(fc7,self.KEEP_PROB)
        
        """
            构建输出层
        """
        self.fc8=fc(dropout7,4096,self.NUM_CLASSES,relu=False,name='logits')
        
    def load_initial_weights(self,session):
        """
            加载权重和偏置数据
        """
        weights_dict=np.load(self.WEIGHTS_PATH,encoding='bytes').item()
        
        for op_name in weights_dict:

            # Check if layer should be trained from scratch
            if op_name not in self.SKIP_LAYER:

                with tf.variable_scope(op_name, reuse=True):

                    # Assign weights/biases to their corresponding tf variable
                    for data in weights_dict[op_name]:

                        # Biases
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=False)
                            session.run(var.assign(data))

                        # Weights
                        else:
                            var = tf.get_variable('weights', trainable=False)
                            session.run(var.assign(data))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
