# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:35:42 2020
@author: Dreamen
"""
from __future__ import print_function
import numpy as np
import warnings
import keras
from keras import layers
from keras.layers import Input,Dense,Activation,Flatten,Conv2D,MaxPooling2D,GlobalMaxPooling2D,ZeroPadding2D
from keras.layers import GlobalAveragePooling2D,AveragePooling2D,BatchNormalization,Lambda
from keras.models import Model
from keras.preprocessing import image
from keras.regularizers import l1,l2,l1_l2
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
# import tensorflow as tf
# tf.test.gpu_device_name()

def L12_reg(weight_matrix):
    return None# 0.01 * K.sum(K.abs(weight_matrix)) + 0.000001 * K.sum(K.pow(weight_matrix,2))



def shortcut(input, residual):
    """
    shortcut连接，也就是identity mapping部分。
    """
 
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_height = int(round(input_shape[1] / residual_shape[1]))
    stride_width = int(round(input_shape[2] / residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3]
 
    identity = input
    # 如果维度不同，则使用1x1卷积进行调整
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        identity = Conv2D(filters=residual_shape[3],
                           kernel_size=(1, 1),
                           strides=(stride_width, stride_height),
                           padding="valid",
                           kernel_regularizer=L12_reg)(input)
 
    return layers.add([identity, residual])
#原文链接：https://blog.csdn.net/zzc15806/article/details/83540661
def identity_block(input_tensor, kernel_size, filters, stage, block,use_bias=True, train_bn=True, strides=(1, 1)):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    stage is phase for different
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    x = Conv2D(filters1, strides =strides, kernel_size=(3, 3), name=conv_name_base + '2a', kernel_initializer="he_normal",
                      padding='same',kernel_regularizer=L12_reg)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, strides = (1, 1),kernel_size=(3, 3), kernel_initializer="he_normal",
               padding='same', name=conv_name_base + '2b',kernel_regularizer=L12_reg)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = shortcut(input_tensor,x)


    # x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c', kernel_initializer="he_normal",
    #                   kernel_regularizer=L12_reg)(x)
    # x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    # x = layers.add([x, input_tensor])
    # x = Activation('relu')(x)
    return x
#%%
def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), use_bias=True, train_bn=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """

    filters1, filters2, filters3 = filters # size not number

    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,kernel_regularizer=L12_reg, kernel_initializer="he_normal",
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', kernel_initializer="he_normal",
               name=conv_name_base + '2b',kernel_regularizer=L12_reg)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)# the relu function used for the subsection and input is the sum of

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c', kernel_initializer="he_normal",
                      kernel_regularizer=L12_reg)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, kernel_regularizer=L12_reg, kernel_initializer="he_normal",
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x
#%%
def resnet_self(input_tensor = None, include_top=True,num_outputs=1,
                 input_shape=(224,224,3), architecture = 'resnet18', stage5=True, train_bn=True):
    """Build a ResNet graph.
        architecture: Can be resnet50 or resnet101
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    img_input = Input(shape=input_shape,name = 'input')

    # assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    # x = KL.ZeroPadding2D((3, 3))(img_input)
    c01=x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True, kernel_initializer="he_normal",
                      padding = 'same',kernel_regularizer=L12_reg)(img_input)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2  block=3
    # x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    c12= x = identity_block(x, 3,[64, 64, 64], strides=(1, 1), stage=2, block='b', train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 64], strides=(1, 1),stage=2, block='c', train_bn=train_bn)
    # Stage 3
    # x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
    # x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)# at shortcut no conv see notebook
    c23= x = identity_block(x, 3,[128, 128, 128],  strides=(2, 2),stage=3, block='c', train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 128],strides=(1, 1), stage=3, block='d', train_bn=train_bn)
    # Stage 4
    # x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
    block_count = {"resnet18": 2,"resnet50": 5, "resnet101": 22}[architecture] # if architecture is resnet50, the block_count is 5
    for i in range(block_count):
        if i == 0:
            c34 = x = identity_block(x, 3,[256, 256, 256],  strides=(2, 2),stage=4, block=chr(98 + i), train_bn=train_bn)#chr(98) is b
        else:
            x = identity_block(x, 3,[256, 256, 256],  strides=(1, 1),stage=4, block=chr(98 + i), train_bn=train_bn)
    C4 = x
    # Stage 5
    if stage5:
        # x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
        c45 = x = identity_block(x, 3, [512, 512, 512], strides=(2, 2),stage=5, block='b', train_bn=train_bn)
        C5 = x = identity_block(x, 3, [512, 512, 512], stage=5, block='c', train_bn=train_bn)
    else:
        C5 = None
    # _, C2, C3, C4, C5 = resnet_graph(input_image=None, architecture="resnet50",stage5=True, train_bn=True)
# Top-down Layers 构建自上而下的网络结构
# 从 C5开始处理，先卷积来转换特征图尺寸

    P5 = KL.Conv2D(256, (1, 1), name='fpn_c5p5',kernel_regularizer=L12_reg)(C5)  # 256
    P4 = KL.Add(name="fpn_p5addc4")([
                KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
                KL.Conv2D(256, (1, 1),name='fpn_c4p4')(C4)])
    x = KL.Activation('relu')(P4)
    x = BatchNormalization(axis=3, name='bnp4')(x)
    P4 = Activation('relu')(x)
    P4 = KL.Conv2D(256, (3, 3), padding="SAME", kernel_initializer="he_normal",name="fpn_p4")(P4)
    P3 = KL.Add(name="fpn_p4addc3")([
                KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
                KL.Conv2D(256, (1, 1), name='fpn_c3p3')(C3)])

    x = KL.Activation('relu')(P3)
    x = BatchNormalization(axis=3, name='bnp3')(x)
    P3 = Activation('relu')(x)
    P3 = KL.Conv2D(256, (3, 3), padding="SAME",name="fpn_p3", kernel_initializer="he_normal")(P3)
    P2 = KL.Add(name="fpn_p3addc2")([
                KL.UpSampling2D(size=(2, 2),name="fpn_p3upsampled")(P3),
                KL.Conv2D(256, (1, 1),  kernel_initializer="he_normal",name='fpn_c2p2')(C2)])
    x = KL.Activation('relu')(P2)
    x = BatchNormalization(axis=3, name='bnp2')(x)
    P2 = Activation('relu')(x)
    P2 = KL.Conv2D(256, (3, 3), padding="SAME", kernel_initializer="he_normal", name="fpn_p2")(P2)
    #原始fpn50没有p1
    # P1 = KL.Add(name = "fpn_p2addc1")([P2,KL.Conv2D(256, (1, 1), name='fpn_c1p1')(C1)])# change channel 64 to 256 for C1
    # P2-P5最后又做了一次3*3的卷积，作用是消除上采样带来的混叠效应
    # Attach 3x3 conv to all P layers to get the final feature maps.
    # P1 = KL.Conv2D(256, (3, 3), padding="SAME",  kernel_initializer="he_normal",name="fpn_p1")(P1)
    P2 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p2conved")(P2)
    P3 = KL.Conv2D(256, (3, 3), padding="SAME",name="fpn_p3conved")(P3)
    P4 = KL.Conv2D(256, (3, 3), padding="SAME",name="fpn_p4conved")(P4)
    P5 = KL.Conv2D(256, (3, 3), padding="SAME",name="fpn_p5conved")(P5)

    # rpn_feature_maps = [P2, P3, P4, P5, P6]
    #C1 out 55*55*64 the channel of C1 is the same as the C2, so no pooling
    # C12_input = KL.Conv2D(256, (1, 1), name='fpn_C1toC2')(C1)
    # C12_input = Activation('relu')(C12_input)
    #c2 out 55*55*256    C1 output 56*56*64
    #525 原始没有融合c1去掉
    '''
    C2_input = KL.Conv2D(256, (1, 1), name='fpn_C1toC2')(C1)
    C2_output = KL.Add(name="fpn_C1addC2")([C2,C2_input])
    '''
    # stage 3
    pool2 = MaxPooling2D(pool_size=(2, 2))(C2)
    C3_input = KL.Conv2D(128, (1, 1), name='fpn_C2toC3',kernel_regularizer=L12_reg)(pool2)
    #c3 out 28*28*512
    C3_output = KL.Add(name="fpn_C2addC3")([C3,C3_input])
    #stage 4
    pool34 = MaxPooling2D(pool_size=(2, 2))(C3_output)
    C4_input = KL.Conv2D(256, (1, 1),  kernel_initializer="he_normal",name='fpn_C3toC4')(pool34)
    #c4 out 14*14*1024
    C4_output = KL.Add(name="fpn_C3addC4")([C4,C4_input])
    # stage 5 C5 output 7*7*2048
    pool45 = MaxPooling2D(pool_size=(2, 2))(C4_output)
    C5_input = KL.Conv2D(512, (1, 1),  kernel_initializer="he_normal",name='fpn_C4toC5')(pool45)
    #c5 out output 7*7*2048
    C5_output = KL.Add(name="fpn_C4addC5")([C5,C5_input])
    # C5_output = layers.add([C5, C5_input])
    gpC1 = GlobalAveragePooling2D(dim_ordering='default', name='global_pool1')(C1)
    gpC2 = GlobalAveragePooling2D(dim_ordering='default', name='global_pool2')(C2)
    gpC3 = GlobalAveragePooling2D(dim_ordering='default', name='global_pool3')(C3_output)
    gpC4 = GlobalAveragePooling2D(dim_ordering='default', name='global_pool4')(C4_output)

    gpC5 = GlobalAveragePooling2D(dim_ordering='default', name='global_pool5')(C5_output)
    gpCall= [gpC1,gpC2,gpC3,gpC4,gpC5]
    featureall = KL.concatenate(gpCall)
    featureall2 = Dense(64, activation='relu', name='Dense1')(featureall)
    # the gradient for main task if not allowed back propagate to the subtask
    '''
    stop_grad = Lambda(lambda x: K.stop_gradient(x))(featureall2)
    featureall1 = KL.Dropout(0.5)(featureall2)
    output_substage = Dense(3, activation='softmax', name='TNM_stage')(featureall1)#loss="categorical_crossentropy"
    '''
    gbpooling6 = KL.GlobalAveragePooling2D(dim_ordering='default', name='global_pool6')(P2)
    merge1 = KL.concatenate([gbpooling6,featureall2])#if is multi-task, replace featureall2 by stop_grad
    #525 将下面输入merge1 改为gbpooling6
    output1 = KL.Dense(64,activation='relu', name='Dense2')(gbpooling6)
    # output1 = KL.Dropout(0.5,name = 'dropout')(output1)
    output1 = KL.Dense(num_outputs,activation='sigmoid', name='risk_pred')(output1)
    # input_shape = (224,224,3)
    # img_input = Input(shape=input_shape,name = 'input')
    model50 = Model(inputs = img_input, outputs =  output1, name='model50')
    return model50
#%%
if __name__ == '__main__':
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from keras.utils import plot_model
    model50 = resnet_self(include_top=True)
    plot_model(model50,to_file='./resnet18FPN525removeCbranch.pdf',show_shapes=True)
