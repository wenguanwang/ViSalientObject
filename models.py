from __future__ import division
from keras.layers import MaxPooling2D, UpSampling2D, Concatenate
from keras.layers.convolutional import Conv2D
def ss_vgg(data):
    # conv_1
    trainable = True
    conv_1_out = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=trainable)(data)
    conv_1_out = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=trainable)(conv_1_out)
    ds_conv_1_out = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(conv_1_out)

    # conv_2
    conv_2_out = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=trainable)(ds_conv_1_out)
    conv_2_out = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=trainable)(conv_2_out)
    ds_conv_2_out = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(conv_2_out)

    # conv_3
    conv_3_out = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=trainable)(ds_conv_2_out)
    conv_3_out = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=trainable)(conv_3_out)
    conv_3_out = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=trainable)(conv_3_out)
    ds_conv_3_out = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', padding='same')(conv_3_out)

    # conv_4
    conv_4_out = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=trainable)(ds_conv_3_out)
    conv_4_out = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=trainable)(conv_4_out)
    conv_4_out = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=trainable)(conv_4_out)
    ds_conv_4_out = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', padding='same')(conv_4_out)

    # conv_5 #
    conv_5_out = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=trainable)(ds_conv_4_out)
    conv_5_out = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=trainable)(conv_5_out)
    conv_5_out = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=trainable)(conv_5_out)
    # conv_5_out = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', padding='same')(conv_5_out)
    saliency_conv_5 = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='sal_conv_5', trainable=trainable)(conv_5_out)

    conv_4_out = Conv2D(64, (3, 3), padding='same', activation='sigmoid', name='conv_4_out', trainable=trainable)(conv_4_out)
    up_saliency_conv_5 = UpSampling2D(size=(2, 2))(saliency_conv_5)
    conv_4_out = Concatenate()([conv_4_out, up_saliency_conv_5])
    saliency_conv_4 = Conv2D(1, (3, 3), padding='same', activation='sigmoid', name='sal_conv4', trainable=trainable)(conv_4_out)

    # saliency from conv_3 #
    conv_3_out = Conv2D(64, (3, 3), padding='same', activation='sigmoid', name='conv_3_out', trainable=trainable)(conv_3_out)#, activation='sigmoid'
    up_saliency_conv_4 = UpSampling2D(size=(2, 2))(saliency_conv_4)
    conv_3_out = Concatenate()([conv_3_out, up_saliency_conv_4])
    saliency_conv_3 = Conv2D(1, (3, 3), padding='same', activation='sigmoid', name='sal_conv3', trainable=trainable)(conv_3_out)

    return [saliency_conv_5, saliency_conv_4, saliency_conv_3]#

def ds_vgg(data):
    # conv_1
    trainable = True
    data = Concatenate()([data[0], data[1], data[2]])
    conv_1_out = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1_new', trainable=trainable)(data)
    conv_1_out = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=trainable)(
        conv_1_out)
    ds_conv_1_out = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(conv_1_out)

    # conv_2
    conv_2_out = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=trainable)(
        ds_conv_1_out)
    conv_2_out = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=trainable)(
        conv_2_out)
    ds_conv_2_out = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(conv_2_out)

    # conv_3
    conv_3_out = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=trainable)(
        ds_conv_2_out)
    conv_3_out = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=trainable)(
        conv_3_out)
    conv_3_out = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=trainable)(
        conv_3_out)
    ds_conv_3_out = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', padding='same')(conv_3_out)

    # conv_4
    conv_4_out = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=trainable)(
        ds_conv_3_out)
    conv_4_out = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=trainable)(
        conv_4_out)
    conv_4_out = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=trainable)(
        conv_4_out)
    ds_conv_4_out = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', padding='same')(conv_4_out)

    # conv_5 #
    conv_5_out = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=trainable)(
        ds_conv_4_out)
    conv_5_out = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=trainable)(
        conv_5_out)
    conv_5_out = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=trainable)(
        conv_5_out)
    # conv_5_out = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', padding='same')(conv_5_out)
    saliency_conv_5 = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='sal_conv_5', trainable=trainable)(
        conv_5_out)

    conv_4_out = Conv2D(64, (3, 3), padding='same', activation='sigmoid', name='conv_4_out', trainable=trainable)(
        conv_4_out)
    up_saliency_conv_5 = UpSampling2D(size=(2, 2))(saliency_conv_5)
    conv_4_out = Concatenate()([conv_4_out, up_saliency_conv_5])
    saliency_conv_4 = Conv2D(1, (3, 3), padding='same', activation='sigmoid', name='sal_conv4', trainable=trainable)(
        conv_4_out)

    # saliency from conv_3 #
    conv_3_out = Conv2D(64, (3, 3), padding='same', activation='sigmoid', name='conv_3_out', trainable=trainable)(
        conv_3_out)  # , activation='sigmoid'
    up_saliency_conv_4 = UpSampling2D(size=(2, 2))(saliency_conv_4)
    conv_3_out = Concatenate()([conv_3_out, up_saliency_conv_4])
    saliency_conv_3 = Conv2D(1, (3, 3), padding='same', activation='sigmoid', name='sal_conv3', trainable=trainable)(
        conv_3_out)

    return [saliency_conv_5, saliency_conv_4, saliency_conv_3]#