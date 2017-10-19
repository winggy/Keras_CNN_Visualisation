
import os
from glob import glob
import numpy as np
import cv2
from keras.models import load_model
from vis.utils import utils
from keras import activations
from vis.visualization import overlay, visualize_cam
import scipy.misc


if __name__ == '__main__':

    IMG_SIZE = 256
    CHANNELS = 1

    input_path = r'C:\Users\apiccinnano\Documents\MATLAB\Attacks_JPEG_CNN\Results'

    # Load the trained CNN model
    modelpath = r'C:\Users\apiccinnano\PycharmProjects\TensorflowTests\single-vs-double_keras_{}x{}\trainedmodel\snapshots\final_trained_model.h5'.format(IMG_SIZE,IMG_SIZE)
    model = load_model(modelpath)
    original_model = model

    # Search for layer index by name. Alternatively, set this to -1 since it corresponds to the last layer
    layer_idx = utils.find_layer_idx(model, 'predictions')

    # Swap SoftMax with linear: it provides better visualisation results
    model.layers[layer_idx].activation = activations.linear

    # To update model, there is a trick: save it and reload it
    model.save('tmp.h5')
    model = load_model('tmp.h5')
    os.remove('tmp.h5')

    # For each folder
    for f in glob(input_path+'/*/'):

        print(os.path.basename(os.path.normpath(f)))

        for p in np.arange(0.0, 1.1, 0.1):

            mask_name = 'mask_single_{:.1f}_double_{:.1f}.png'.format(1-p, p)
            img_name = 'single_{:.1f}_double_{:.1f}.bmp'.format(1-p, p)

            print(mask_name)
            print(img_name)

            mask = cv2.imread(os.path.join(f, mask_name, cv2.IMREAD_GRAYSCALE))
            test_img = cv2.imread(os.path.join(f, img_name, cv2.IMREAD_GRAYSCALE))

            # From BGR to RGB and scaled
            #test_img = test_img[...,::-1]/255
            #test_img = test_img / 255.

            x = np.zeros((1, IMG_SIZE, IMG_SIZE, CHANNELS))
            x[0] = np.reshape(test_img/255, (IMG_SIZE, IMG_SIZE, CHANNELS))

            class_prob = model.predict(x, verbose=1)
            class_label = model.predict_classes(x, verbose=1)

            print(class_label)

            heatmap = visualize_cam(model,
                                    layer_idx,
                                    filter_indices=0,
                                    seed_input=x[0],
                                    backprop_modifier='guided')

            if x[0].shape[2] < 3:
                overlayed_map = overlay(np.tile(x[0], (1, 1, 3)), heatmap)

            rgbmask = np.uint8(np.tile(np.reshape(mask, (IMG_SIZE,IMG_SIZE,1)), (1, 1, 3)))
            rgb_input = np.uint8(np.tile(np.reshape(test_img, (IMG_SIZE, IMG_SIZE, 1)), (1, 1, 3)))

            res = np.hstack((rgbmask, rgb_input, overlayed_map))
            #cv2.namedWindow('test')
            #cv2.imshow('test', res)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            scipy.misc.imsave('detection_single_{:.1f}_double_{:.1f}.png'.format(1-p, p), scipy.misc.imresize(res,100))

