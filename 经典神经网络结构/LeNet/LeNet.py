# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 21:24:30 2018

@author: yupeifeng

"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
tf.logging.set_verbosity(tf.logging.INFO)

def lenet_model_fn(features,labels,mode):
    """
        首先定义一个输入层
        把训练集X通过reshape变化变为4维的Tensor : [batch_size,width,height,channels]
        MNIST的训练图片是28*28像素的，而且只有一个颜色通道
        但是在LeNet模型中其定义输入的数据图片的是32*32*C
    """
    features["x"]=tf.reshape(features["x"],[-1,28,28,1])
    input_layer=tf.pad(features["x"],((0,0),(2,2),(2,2),(0,0)))
    
    """
        构建第一层卷积神经网络
        Input Tensor Shape : [batch_size,32,32,1]
        用6个5*5的filter进行卷积操作
        stride=1
        padding="valid"
        Output Tensor Shape : [batch_size,28,28,6]
    """
    conv1=tf.layers.conv2d(
            inputs=input_layer,
            filters=6,
            kernel_size=[5,5],
            strides=(1,1),
            padding="valid",
            activation=tf.nn.relu
            )
    
    """
        构建第一个池化池
        使用max池化层，fitler为2*2，strides=2
        Input Tensor Shape : [batch_size,28,28,6]
        Output Tensor Shape : [batch_size,14,14,6]
    """
    pool1=tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],strides=2)
    
    """
        构建第二个卷积层
        Input Tensor Shape : [batch_size,14,14,6]
        用16个5*5的filter
        padding="valid"
        Output Tensor Shape : [batch_size,10,10,16]
    """
    conv2=tf.layers.conv2d(
            inputs=pool1,
            filters=16,
            kernel_size=[5,5],
            strides=(1,1),
            padding="valid",
            activation=tf.nn.relu
            )
    """
        构建第二个池化层
        使用max池化层，filter为2*2，strides=2
        Input Tensor Shape : [batch_size,10,10,16]
        Output Tensor Shape : [batch_size,5,5,16]
    """
    pool2=tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=2)
    
    """
        构建第一个全连接层
        Input Tensor Shape : [batch_size,5,5,16]
        Output Tensor Shape : [batch_size,5*5*16]
    """
    fc0=tf.reshape(pool2,[-1,5*5*16])
    
    """
        构建第一个全连接层
        全连接层有120个神经元
        Input Tensor Shape : [batch_size,5*5*16]
        Output Tensor Shape : [batch_size,120]
    """
    fc1=tf.layers.dense(inputs=fc0,units=120,activation=tf.nn.relu)
    
    """
        构建第二个全连接层
        全连接层有84个神经元
        Input Tensor Shape : [batch_size,120]
        Output Tensor Shape : [batch_size,84]
    """
    fc2=tf.layers.dense(inputs=fc1,units=84,activation=tf.nn.relu)
    
    """
        构建输出层
        Input Tensor Shape : [batch_size,84]
        Output Tensor Shape : [batch_size,10]
    """
    logits=tf.layers.dense(inputs=fc2,units=10)
    
    predictions={
            #生成预测
            "classes" : tf.argmax(input=logits,axis=1),
            #在计算图中添加softmax_tensor
            "probabilities" : tf.nn.softmax(logits,name="softmax_tensor")
            }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)
    
    """
        计算损失
    """
    loss=tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)
    
    """
        定义训练操作
    """
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer=tf.train.AdamOptimizer(learning_rate=0.001)
        train_op=optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step()
                )
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)
    
    """
        添加评估操作
    """

    eval_metric_ops={
            "accuracy" : tf.metrics.accuracy(labels,predictions=predictions["classes"])
            }
    return tf.estimator.EstimatorSpec(
            mode=mode,loss=loss,eval_metric_ops=eval_metric_ops
            )
    
    
def main(unused_argv):
    """
        加载训练集和评估集
    """    
    mnist=tf.contrib.learn.datasets.load_dataset("mnist")
    train_data=mnist.train.images
    train_labels=np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    
    """
        构建一个Estimator
    """
    mnist_classifier=tf.estimator.Estimator(
            model_fn=lenet_model_fn,
            model_dir="models/lenet"
            )
    """
        每隔50步，进行记录
    """
    
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)
    
    """
        构建训练过程
    """
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=128,
            num_epochs=None,
            shuffle=True)
    mnist_classifier.train(
            input_fn=train_input_fn,
            steps=20000,
            hooks=[logging_hook]
            )
    
    """
        构建测试过程
    """
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            y=eval_labels,
            num_epochs=1,
            shuffle=False
            )
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    
    print(eval_results)
    
    
    
if __name__ == "__main__":
    tf.app.run()
    

