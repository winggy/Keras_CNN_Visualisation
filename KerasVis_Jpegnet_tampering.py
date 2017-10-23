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

    input_path = r'C:\Users\utente\PycharmProjects\FiltersVisualisation_Keras\Matlab_Results'

    # Load the trained CNN model
    modelpath = r'C:\Users\utente\PycharmProjects\TensorflowTests\single-vs-double_keras_{}x{}\trainedmodel\snapshots\final_trained_model.h5'.format(IMG_SIZE,IMG_SIZE)
    model = load_model(modelpath)
    original_model = model

    # Search for layer index by name. Alternatively, set this to -1 since it corresponds to the last layer
    layer_idx = utils.find_layer_idx(model, 'predictions')

    # Swap SoftMax with linear: it provides better visualisation results
    model.layers[layer_idx].activation = activations.linear

    orig_model = model

    # To update model, there is a trick: save it and reload it
    model.save('tmp.h5')
    model = load_model('tmp.h5')
    os.remove('tmp.h5')

    # For each folder
    for f in glob(input_path+'/*/'):

        current_folder = os.path.basename(os.path.normpath(f))

        for p in np.arange(0.0, 1.1, 0.1):

            mask_name = 'mask_single_{:.1f}_double_{:.1f}.png'.format(1-p, p)
            img_name = 'single_{:.1f}_double_{:.1f}.bmp'.format(1-p, p)

            print(mask_name)
            print(img_name)

            mask = cv2.imread(os.path.join(f, mask_name), cv2.IMREAD_GRAYSCALE)
            test_img = cv2.imread(os.path.join(f, img_name), cv2.IMREAD_GRAYSCALE)

            # From BGR to RGB and scaled
            #test_img = test_img[...,::-1]/255
            #test_img = test_img / 255.

            x = np.zeros((1, IMG_SIZE, IMG_SIZE, CHANNELS))
            x[0] = np.reshape(test_img/255, (IMG_SIZE, IMG_SIZE, CHANNELS))

            class_prob = orig_model.predict(x, verbose=1)
            class_label = orig_model.predict_classes(x, verbose=1)

            class_name = 'single' if class_label == 1 else 'double'

            print(class_name)

            heatmap_double = visualize_cam(model,
                                          layer_idx,
                                          filter_indices=0,
                                          seed_input=x[0],
                                          backprop_modifier='guided')

            heatmap_single = visualize_cam(model,
                                          layer_idx,
                                          filter_indices=1,
                                          seed_input=x[0],
                                          backprop_modifier='guided')

            if x[0].shape[2] < 3:
                overlayed_map_s = overlay(np.tile(x[0], (1, 1, 3)), heatmap_single)

            if x[0].shape[2] < 3:
                overlayed_map_d = overlay(np.tile(x[0], (1, 1, 3)), heatmap_double)

            rgbmask = np.uint8(np.tile(np.reshape(mask, (IMG_SIZE,IMG_SIZE,1)), (1, 1, 3)))
            rgb_input = np.uint8(np.tile(np.reshape(test_img, (IMG_SIZE, IMG_SIZE, 1)), (1, 1, 3)))

            res = np.hstack((rgbmask, rgb_input, overlayed_map_s, overlayed_map_d))
            #cv2.namedWindow('test')
            #cv2.imshow('test', res)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            if not os.path.exists('detection/'+current_folder):
                os.makedirs('detection/'+current_folder)

            class_name = 'single' if class_label == 1 else 'double'
            scipy.misc.imsave('detection/{}/single_{:.1f}_double_{:.1f}_out={}.png'.format(current_folder, 1-p, p, class_name), scipy.misc.imresize(res,100))
