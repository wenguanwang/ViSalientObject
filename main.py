from __future__ import division
from keras.layers import Input
from keras.models import Model
import os
import sys
import numpy as np
from config import *
from utilities import postprocess_predictions
from models import ss_vgg, ds_vgg
from scipy.misc import imread, imsave, imresize

def get_test(data):
    Xims_224 = np.zeros((1, 224, 224, 3))
    img = imread(data['image'])
    img_name = os.path.basename(data['image'])
    if img.ndim == 2:
        copy = np.zeros((img.shape[0], img.shape[1], 3))
        copy[:, :, 0] = img
        copy[:, :, 1] = img
        copy[:, :, 2] = img
        img = copy
    r_img = imresize(img, (224, 224))
    r_img = r_img.astype(np.float32)
    r_img[:, :, 0] -= img_channel_mean[0]
    r_img[:, :, 1] -= img_channel_mean[1]
    r_img[:, :, 2] -= img_channel_mean[2]
    r_img = r_img[:, :, ::-1]
    Xims_224[0, :] = np.copy(r_img)
    return Xims_224, img, img_name


def get_dynamic_test(data):
    X1ims_224 = np.zeros((1, 224, 224, 3))
    X2ims_224 = np.zeros((1, 224, 224, 3))
    X3ims_224 = np.zeros((1, 224, 224, 1))
    img1 = imread(data['image1'])
    img2 = imread(data['image2'])
    static_saliencymap = imread(data['static_saliency1'])
    img_name = os.path.basename(data['image1'])
    if img1.ndim == 2:
        copy = np.zeros((img1.shape[0], img1.shape[1], 3))
        copy[:, :, 0] = img1
        copy[:, :, 1] = img1
        copy[:, :, 2] = img1
        img1 = copy
        copy[:, :, 0] = img2
        copy[:, :, 1] = img2
        copy[:, :, 2] = img2
        img2 = copy

    r_img1 = imresize(img1, (224, 224))
    r_img1 = r_img1.astype(np.float32)
    r_img1[:, :, 0] -= img_channel_mean[0]
    r_img1[:, :, 1] -= img_channel_mean[1]
    r_img1[:, :, 2] -= img_channel_mean[2]
    r_img1 = r_img1[:, :, ::-1]
    X1ims_224[0, :] = np.copy(r_img1)

    r_img2 = imresize(img2, (224, 224))
    r_img2 = r_img2.astype(np.float32)
    r_img2[:, :, 0] -= img_channel_mean[0]
    r_img2[:, :, 1] -= img_channel_mean[1]
    r_img2[:, :, 2] -= img_channel_mean[2]
    r_img2 = r_img2[:, :, ::-1]
    X2ims_224[0, :] = np.copy(r_img2)

    static_saliencymap_224 = imresize(static_saliencymap, (224, 224))
    static_saliencymap_224 = static_saliencymap_224.astype(np.float32)

    X3ims_224[0, :, :, 0] = np.copy(static_saliencymap_224)
    return [X1ims_224, X2ims_224, X3ims_224], img1, img_name

if __name__ == '__main__':
    if len(sys.argv) != 1:
        raise NotImplementedError
    else:
        extra_video_saliency = './DAVIS/'  # testing/
        video_paths = [dataset for dataset in os.listdir(extra_video_saliency + 'video/') if
                        os.path.isdir(extra_video_saliency + 'video/')]

        x = Input(batch_shape=(1, 224, 224, 3))
        static_m = Model(inputs=x, outputs=ss_vgg(x))
        print("Loading weights of static saliency model ")
        static_m.load_weights('static_vgg.h5')
        print("Making static saliency prediction")

        for videos_path in video_paths:
            video_output = './DAVIS/static_saliency/' + videos_path
            if not os.path.exists(video_output):
                os.makedirs(video_output)
            # prepare test data
            images = [extra_video_saliency + 'video/' + videos_path + '/' + f
                        for f in os.listdir(extra_video_saliency + 'video/' + videos_path) if
                        f.endswith(('.jpg', '.jpeg', '.png'))]
            images.sort()
            test_data = []
            for image in images:
                annotation_data = {'image': image}
                test_data.append(annotation_data)
            # compute static saliency
            for data in test_data:
                Ximg, original_image, img_name = get_test(data)
                predictions = static_m.predict(Ximg, batch_size=1)
                static_saliency = postprocess_predictions(predictions[2][0, :, :, 0], original_image.shape[0],
                                                               original_image.shape[1])
                imsave(video_output + '/%s.png' % img_name[0:-4], static_saliency.astype(int))

        x1 = Input(batch_shape=(1, 224, 224, 3))
        x2 = Input(batch_shape=(1, 224, 224, 3))
        x3 = Input(batch_shape=(1, 224, 224, 1))
        dynamic_m = Model(inputs=[x1, x2, x3], outputs=ds_vgg([x1, x2, x3]))
        print("Loading weights of dynamic saliency model")
        dynamic_m.load_weights('dynamic_vgg.h5')
        print("Making dynamic saliency prediction")
        for videos_path in video_paths:
            video_output = 'DAVIS/dynamic_saliency/' + videos_path
            if not os.path.exists(video_output):
                os.makedirs(video_output)

            # prepare test data
            images1 = [extra_video_saliency + 'video/' + videos_path + '/' + f
                        for f in os.listdir(extra_video_saliency + 'video/' + videos_path)[0:-1] if
                        f.endswith(('.jpg', '.jpeg', '.png'))]

            images2 = [extra_video_saliency + 'video/' + videos_path + '/' + f
                        for f in os.listdir(extra_video_saliency + 'video/' + videos_path)[1:] if
                        f.endswith(('.jpg', '.jpeg', '.png'))]

            static_saliencys1 = [extra_video_saliency + 'static_saliency/' + videos_path + '/' + f
                                    for f in os.listdir(extra_video_saliency + 'static_saliency/' + videos_path)[0:-1] if
                                    f.endswith(('.jpg', '.jpeg', '.png'))]

            images1.sort()
            images2.sort()
            static_saliencys1.sort()
            test_data = []

            for image1, image2, static_saliency1 in zip(images1, images2, static_saliencys1):
                annotation_data = {'image1': image1, 'image2': image2, 'static_saliency1': static_saliency1}
                test_data.append(annotation_data)

            # compute dynamic saliency
            for data in test_data:
                Ximg, original_image, img_name = get_dynamic_test(data)
                predictions = dynamic_m.predict(Ximg, batch_size=1)
                static_saliency = postprocess_predictions(predictions[2][0, :, :, 0], original_image.shape[0],
                                                               original_image.shape[1])
                imsave(video_output + '/%s.png' % img_name[0:-4], static_saliency.astype(int))
