import os
from glob import glob
import numpy as np
import cv2
from keras.models import load_model
from vis.utils import utils
from keras import activations
from vis.visualization import overlay, visualize_cam
import scipy.misc
from utils import draw_captions


if __name__ == '__main__':

    # Patch size (IMG_SIZE x IMG_SIZE)
    IMG_SIZE = 256
    
    # Input patch channels 
    CHANNELS = 1

    # Retrieve double compressed patches generated with Matlab's JPEG and denoising here. Here, for each folder, 
    # there are 10 versions of the same image with increasing percentage (+10%) of double JPEG compressed pixels
    input_path = r'C:\Users\utente\PycharmProjects\FiltersVisualisation_Keras\Matlab_Results'

    # ---------------------------------------------------------------------------------------------------------------------
    # MODEL PREPARATION
    # ---------------------------------------------------------------------------------------------------------------------
    
    # Load the trained CNN model
    modelpath = r'C:\Users\utente\PycharmProjects\TensorflowTests\single-vs-double_keras_{}x{}\trainedmodel\snapshots\final_trained_model.h5'.format(IMG_SIZE,IMG_SIZE)
    model = load_model(modelpath)
    
    # Backup original model before modifying final layer
    original_model = model

    # Search for layer index by name. Alternatively, set this to -1 since it corresponds to the last layer
    layer_idx = utils.find_layer_idx(model, 'predictions')

    # Swap SoftMax with linear: it provides better visualisation results
    model.layers[layer_idx].activation = activations.linear

    # To update model, there is a trick: save it and reload it
    model.save('tmp.h5')
    model = load_model('tmp.h5')
    os.remove('tmp.h5')

    # --------------------------------------------------------------------------------------------------------------
    # LOCALISATION
    # --------------------------------------------------------------------------------------------------------------
    
    # For each folder (representing an image)
    for f in glob(input_path+'/*/'):

        current_folder = os.path.basename(os.path.normpath(f))

        # For each tampered composite image
        for p in np.arange(0.0, 1.1, 0.1):

            mask_name = 'mask_single_{:.1f}_double_{:.1f}.png'.format(1-p, p)
            img_name = 'single_{:.1f}_double_{:.1f}.bmp'.format(1-p, p)

            print('Processing: image {}: {}% single, {}% double'.format(img_name, 1-p, p))

            # Load ground-truth tampering mask and tampered image
            mask = cv2.imread(os.path.join(f, mask_name), cv2.IMREAD_GRAYSCALE)
            test_img = cv2.imread(os.path.join(f, img_name), cv2.IMREAD_GRAYSCALE)

            # From BGR to RGB and scaled
            #test_img = test_img[...,::-1]/255
            #test_img = test_img / 255.

            x = np.zeros((1, IMG_SIZE, IMG_SIZE, CHANNELS))
            x[0] = np.reshape(test_img/255, (IMG_SIZE, IMG_SIZE, CHANNELS))

            # Determine softmax values and class output for current image (use original model)
            class_prob = original_model.predict(x, verbose=1)
            class_label = original_model.predict_classes(x, verbose=1)

            class_name = 'single' if class_label == 1 else 'double'

            print('   > This image is {} with probability={:g}.'.format(class_name.upper, class_prob[class_label]))

            # Localisation of activation regions given class cl = 0 (double)
            heatmap_double = visualize_cam(model,
                                          layer_idx,
                                          filter_indices=0,
                                          seed_input=x[0],
                                          backprop_modifier='guided')
            
            # Localisation of activation regions given class cl = 1 (single)
            heatmap_single = visualize_cam(model,
                                          layer_idx,
                                          filter_indices=1,
                                          seed_input=x[0],
                                          backprop_modifier='guided')

            if x[0].shape[2] < 3:
                overlayed_map_s = overlay(np.tile(x[0], (1, 1, 3)), heatmap_single)
                overlayed_map_d = overlay(np.tile(x[0], (1, 1, 3)), heatmap_double)

            rgbmask = np.uint8(np.tile(np.reshape(mask, (IMG_SIZE,IMG_SIZE,1)), (1, 1, 3)))
            rgb_input = np.uint8(np.tile(np.reshape(test_img, (IMG_SIZE, IMG_SIZE, 1)), (1, 1, 3)))
    
            # Make a composite of input data and heatmaps
            res = np.hstack((rgbmask, rgb_input, overlayed_map_s, overlayed_map_d))

            # Create an output folder with the same name of input folder
            if not os.path.exists('detection/'+current_folder):
                os.makedirs('detection/'+current_folder)

            # Store maps and data
            class_name = 'single' if class_label == 1 else 'double'
            
            out_file = 'detection/{}/single_{:.1f}_double_{:.1f}_out={}.png'.format(current_folder, 1-p, p, class_name)
            
            # Save the composite image, then reload it and put text labels above each sub-image
            scipy.misc.imsave(out_file, res)
            draw_captions(out_file, out_file, labels=['Mask','Composite','g-CAM single','g-CAM double'])
            
            # Save with .mat extension (.txt would be the same). Load in Matlab with
            # x = load(filename, '-ascii')
            np.savetxt('detection/{}/single_{:.1f}_double_{:.1f}_out={}_gradcam_single.mat'.format(current_folder, 1-p, p, class_name), heatmap_single)
            np.savetxt('detection/{}/single_{:.1f}_double_{:.1f}_out={}_gradcam_double.mat'.format(current_folder, 1-p, p, class_name), heatmap_double)
                    
