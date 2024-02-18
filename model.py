# import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from keras.layers import Activation,Conv2D,MaxPooling2D,BatchNormalization,Input,DepthwiseConv2D,add,Dropout,AveragePooling2D,Concatenate
from keras.models import Model                                                          
import keras.backend as K
from keras.layers import Layer,InputSpec
from keras.utils import conv_utils

from keras.models import Model
from keras.layers import Input, MaxPooling2D, Concatenate, BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras import regularizers
from keras.optimizers import *
import keras
from keras import layers


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    smooth = 1. # ゼロ除算回避のための定数
    y_true_flat = tf.reshape(y_true, [-1]) # 1次元に変換
    y_pred_flat = tf.reshape(y_pred, [-1]) # 同様

    y_true_flat =tf.cast(tf.reshape(y_true, [-1]), tf.float32)  # 1次元に変換
    y_pred_flat = tf.cast(tf.reshape(y_pred, [-1]) , tf.float32) # 同様

    tp = tf.reduce_sum(y_true_flat * y_pred_flat) # True Positive
    nominator = 2 * tp + smooth # 分子
    denominator = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) + smooth # 分母
    score = nominator / denominator
    return 1. - score

def tversky_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    alpha = 0.3 # FP、FNの重み
    smooth = 1.0 # ゼロ除算回避のための定数

    y_true_flat =tf.cast(tf.reshape(y_true, [-1]), tf.float32)  # 1次元に変換
    y_pred_flat = tf.cast(tf.reshape(y_pred, [-1]) , tf.float32) # 同様

    tp = tf.reduce_sum(y_true_flat * y_pred_flat) # True Positive
    fp = tf.reduce_sum((1 - y_true_flat) * y_pred_flat) # False Positive
    fn = tf.reduce_sum(y_true_flat * (1 - y_pred_flat)) # False Negative

    score = (tp + smooth)/(tp + alpha * fp + (1-alpha) * fn + smooth) # Tversky
    return 1. - score

def create_conv(input, filters, l2_reg, name):
    x = Conv2D(filters=filters,
               kernel_size=3,               # 論文の指定通り
               activation='relu',           # 論文の指定通り
               padding='same',              # sameにすることでConcatする際にContracting側の出力のCropが不要になる
               kernel_regularizer=regularizers.l2(l2_reg),
               name=name)(input)
    x = BatchNormalization()(x)
    return x

def create_trans(input, filters, l2_reg, name):
    x = Conv2DTranspose(filters=filters,
                        kernel_size=2,      # 論文の指定通り
                        strides=2,          # このストライドにより出力サイズが入力の2倍に拡大されている
                        activation='relu',  # 論文の指定通り
                        padding='same',     # Concat時のCrop処理回避のため
                        kernel_regularizer=regularizers.l2(l2_reg),
                        name=name)(input)
    x = BatchNormalization()(x)
    return x

class BilinearUpsampling(Layer):

    def __init__(self, upsampling=(2, 2), data_format=None, **kwargs):

        super(BilinearUpsampling, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.upsampling = conv_utils.normalize_tuple(upsampling, 2, 'size')
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        height = self.upsampling[0] * \
            input_shape[1] if input_shape[1] is not None else None
        width = self.upsampling[1] * \
            input_shape[2] if input_shape[2] is not None else None
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        return K.tf.image.resize_bilinear(inputs, (int(inputs.shape[1]*self.upsampling[0]),
                                                   int(inputs.shape[2]*self.upsampling[1])))

def xception_downsample_block(x,channels,top_relu=False):
    ##separable conv1
    if top_relu:
        x=Activation("relu")(x)
    x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Conv2D(channels,(1,1),padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    
    ##separable conv2
    x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Conv2D(channels,(1,1),padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    
    ##separable conv3
    x=DepthwiseConv2D((3,3),strides=(2,2),padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Conv2D(channels,(1,1),padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)
    return x

def res_xception_downsample_block(x,channels):
    res=Conv2D(channels,(1,1),strides=(2,2),padding="same",use_bias=False)(x)
    res=BatchNormalization()(res)
    x=xception_downsample_block(x,channels)
    x=add([x,res])
    return x

def xception_block(x,channels):
    ##separable conv1
    x=Activation("relu")(x)
    x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Conv2D(channels,(1,1),padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)
    
    ##separable conv2
    x=Activation("relu")(x)
    x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Conv2D(channels,(1,1),padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)
    
    ##separable conv3
    x=Activation("relu")(x)
    x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Conv2D(channels,(1,1),padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)
    return x
    
def res_xception_block(x,channels):
    res=x
    x=xception_block(x,channels)
    x=add([x,res])
    return x

def aspp(x,input_shape,out_stride):
    b0=Conv2D(256,(1,1),padding="same",use_bias=False)(x)
    b0=BatchNormalization()(b0)
    b0=Activation("relu")(b0)
    
    b1=DepthwiseConv2D((3,3),dilation_rate=(6,6),padding="same",use_bias=False)(x)
    b1=BatchNormalization()(b1)
    b1=Activation("relu")(b1)
    b1=Conv2D(256,(1,1),padding="same",use_bias=False)(b1)
    b1=BatchNormalization()(b1)
    b1=Activation("relu")(b1)
    
    b2=DepthwiseConv2D((3,3),dilation_rate=(12,12),padding="same",use_bias=False)(x)
    b2=BatchNormalization()(b2)
    b2=Activation("relu")(b2)
    b2=Conv2D(256,(1,1),padding="same",use_bias=False)(b2)
    b2=BatchNormalization()(b2)
    b2=Activation("relu")(b2)    

    b3=DepthwiseConv2D((3,3),dilation_rate=(12,12),padding="same",use_bias=False)(x)
    b3=BatchNormalization()(b3)
    b3=Activation("relu")(b3)
    b3=Conv2D(256,(1,1),padding="same",use_bias=False)(b3)
    b3=BatchNormalization()(b3)
    b3=Activation("relu")(b3)
    
    out_shape=int(input_shape[0]/out_stride)
    b4=AveragePooling2D(pool_size=(out_shape,out_shape))(x)
    b4=Conv2D(256,(1,1),padding="same",use_bias=False)(b4)
    b4=BatchNormalization()(b4)
    b4=Activation("relu")(b4)
    b4=BilinearUpsampling((out_shape,out_shape))(b4)
    
    x=Concatenate()([b4,b0,b1,b2,b3])
    return x

def deeplabv3_plus():
    input_shape=(256,256,3)
    input = Input((None,None,3))
    out_stride=16
    num_classes=3

    img_input=Input(shape=input_shape)
    x=Conv2D(32,(3,3),strides=(2,2),padding="same",use_bias=False)(img_input)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    x=Conv2D(64,(3,3),padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    
    x=res_xception_downsample_block(x,128)

    res=Conv2D(256,(1,1),strides=(2,2),padding="same",use_bias=False)(x)
    res=BatchNormalization()(res)    
    x=Activation("relu")(x)
    x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Conv2D(256,(1,1),padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Conv2D(256,(1,1),padding="same",use_bias=False)(x)
    skip=BatchNormalization()(x)
    x=Activation("relu")(skip)
    x=DepthwiseConv2D((3,3),strides=(2,2),padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Conv2D(256,(1,1),padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)    
    x=add([x,res])
    
    x=xception_downsample_block(x,728,top_relu=True)
    
    for i in range(16):
        x=res_xception_block(x,728)

    res=Conv2D(1024,(1,1),padding="same",use_bias=False)(x)
    res=BatchNormalization()(res)    
    x=Activation("relu")(x)
    x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Conv2D(728,(1,1),padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Conv2D(1024,(1,1),padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Conv2D(1024,(1,1),padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)    
    x=add([x,res])
    
    x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Conv2D(1536,(1,1),padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Conv2D(1536,(1,1),padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Conv2D(2048,(1,1),padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)    
    x=Activation("relu")(x)
    
    #aspp
    x=aspp(x,input_shape,out_stride)
    x=Conv2D(256,(1,1),padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    x=Dropout(0.9)(x)
    
    ##decoder 
    x=BilinearUpsampling((4,4))(x)
    dec_skip=Conv2D(48,(1,1),padding="same",use_bias=False)(skip)
    dec_skip=BatchNormalization()(dec_skip)
    dec_skip=Activation("relu")(dec_skip)
    x=Concatenate()([x,dec_skip])
    
    x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    x=Conv2D(256,(1,1),padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    
    x=DepthwiseConv2D((3,3),padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    x=Conv2D(256,(1,1),padding="same",use_bias=False)(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    
    x=Conv2D(num_classes,(1,1),padding="same")(x)
    x=BilinearUpsampling((4,4))(x)
    model=Model(img_input,x)
    return model

def convolution_block(block_input,num_filters=256,kernel_size=3,dilation_rate=1,padding="same",use_bias=False):
    x = layers.Conv2D(num_filters,kernel_size=kernel_size, dilation_rate=dilation_rate, padding="same", use_bias=use_bias)(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)

def ASPP(input):
    dims = input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(input)
    x = convolution_block(x, kernel_size=1, use_bias=True)

    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1x1 = convolution_block(input, kernel_size=1, dilation_rate=1)
    out_rate6 = convolution_block(input, kernel_size=3, dilation_rate=6)
    out_rate12 = convolution_block(input, kernel_size=3, dilation_rate=12)
    out_rate18 = convolution_block(input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1x1, out_rate6, out_rate12, out_rate18])
    output = convolution_block(x, kernel_size=1)

    return output

def deeplabv3_plus_():
    input_shape=(256,256,3)
    input = Input(input_shape)
    out_stride=16
    num_classes=3

    resnet = keras.applications.ResNet50(weights="imagenet", include_top=False, input_tensor=input)
    
    x = resnet.get_layer("conv4_block6_2_relu").output
    x = ASPP(x)

    input_1 = layers.UpSampling2D(size=(input_shape[0] // 4 // x.shape[1], input_shape[0] // 4 // x.shape[2]),interpolation="bilinear",)(x)

    input_2 = resnet.get_layer("conv2_block3_2_relu").output
    input_2 = convolution_block(input_2, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_1, input_2])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(size=(input_shape[0]// x.shape[1], input_shape[0] // x.shape[2]),interpolation="bilinear",)(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    model = keras.Model(inputs=input, outputs=model_output)
    
    return model

loss_object = tf.keras.losses.CategoricalCrossentropy()

def softmax_cross_entropy_loss(y_true, y_pred):
    loss = loss_object(y_true, y_pred)
    return loss

class Custum_class(tf.keras.Model):
    def __init__(self, model, **kwargs):
        super(Custum_class, self).__init__(**kwargs)
        self.model = model

    def compile(self, optimizer, metrics, model_loss_fn, tversky_loss_fn,dice_loss_fn, alpha=0.1, temperature=3, **kwargs):
        super(Custum_class, self).compile(optimizer=optimizer, metrics=metrics, **kwargs)
        self.model_loss = model_loss_fn
        self.tversky_loss = tversky_loss_fn
        self.dice_loss = dice_loss_fn
        self.alpha = alpha
        #self.alpha = alpha
        #self.temperature = temperature

    def train_step(self, data):
        x, y, sample_weight  = tf.keras.utils.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            # 生徒モデルで推定
            cnn_output = self.model(x,training=True)#, training=True)

            # lossを算出
            model_loss = self.model_loss(y, cnn_output )
            tversky_loss = self.tversky_loss(y, cnn_output )
            dice_loss = self.dice_loss(y, cnn_output )

            loss = self.alpha * model_loss + tversky_loss + dice_loss

        # 勾配を算出
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # optimizerでweightを更新
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # compiled_metricsを更新
        y = tf.cast(y, tf.float32)
        self.compiled_metrics.update_state(y,cnn_output)

        # metricsを算出して返す
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"model_loss": model_loss, "tversky_loss": tversky_loss}
        )
        return results

