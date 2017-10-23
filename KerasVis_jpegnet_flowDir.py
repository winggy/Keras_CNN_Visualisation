import os
from vis.visualization import visualize_saliency, overlay, visualize_cam, visualize_activation
from vis.input_modifiers import Jitter
from vis.utils import utils
from keras import activations
from keras.models import load_model
import h5py
import numpy as np
import scipy.misc
import time
from glob import glob


def read_hdf5_compressed(hdf5_file):
    """ Reads training and test data and labels from a compressed HDF5 archive. Returns: (N,H,W,C) tuples of training
        and test data, where N is the number of samples and H,W,C are height, width and channels of the image data;
        (N, Nc) tuples of one-hot train and test labels.
    Keyword arguments:
    hdf5_file       -- full path to the input HDF5 compressed archive
    """

    with h5py.File(hdf5_file, 'r') as hf:
        # Retrieve training data and labels
        g = hf.get('dataset/training')
        training_data = np.array(g.get('data'))
        training_labels = np.array(g.get('labels'))

        # Retrieve test data and labels
        g = hf.get('dataset/test')
        testing_data = np.array(g.get('data'))
        testing_labels = np.array(g.get('labels'))
        hf.close()

    return training_data, testing_data, training_labels, testing_labels


def make_collage(heatmaps, input_img, test_lowpass=[], pxl_margin=5):
    if input_img.shape[2] < 3:
        input_img_rgb = np.tile(input_img, (1, 1, 3))

    if len(test_lowpass.shape) < 3:
        test_lowpass = np.reshape(test_lowpass, test_lowpass.shape + (1,))
        test_lowpass = np.tile(test_lowpass, (1, 1, 3))

    ver_frame = 255 * np.ones((input_img.shape[0], pxl_margin, 3), dtype=heatmaps.dtype)

    collageTop = np.concatenate((ver_frame, input_img_rgb,
                                 ver_frame, heatmaps[0, :, :, 0:3],
                                 ver_frame, heatmaps[0, :, :, 3:6],
                                 #ver_frame, heatmaps[2, :, :, 0:3],
                                 ver_frame), axis=1)

    #collageBottom = np.concatenate((ver_frame, test_lowpass,
    #                                ver_frame, heatmaps[0, :, :, 3:6],
    #                                ver_frame, heatmaps[1, :, :, 3:6],
    #                                ver_frame, heatmaps[2, :, :, 3:6],
    #                                ver_frame), axis=1)

    hor_frame = 255 * np.ones((pxl_margin, collageTop.shape[1], 3))

    #return np.concatenate((hor_frame, collageTop, hor_frame, collageBottom, hor_frame), axis=0)
    return np.concatenate((hor_frame, collageTop, hor_frame), axis=0)


def make_collage_2(heatmaps, input_img, test_lowpass=[], pxl_margin=5):
    if input_img.shape[2] < 3:
        input_img_rgb = np.tile(input_img, (1, 1, 3))
    else:
        input_img_rgb = input_img

    if len(test_lowpass.shape) < 3:
        test_lowpass = np.reshape(test_lowpass, test_lowpass.shape + (1,))
        test_lowpass = np.tile(test_lowpass, (1, 1, 3))

    ver_frame = 255 * np.ones((input_img.shape[0], pxl_margin, 3), dtype=heatmaps.dtype)

    collageTop = np.concatenate((ver_frame, input_img_rgb,
                                 ver_frame, heatmaps[0, :, :, 0:3],
                                 ver_frame), axis=1)

    hor_frame = 255 * np.ones((pxl_margin, collageTop.shape[1], 3))

    return np.concatenate((hor_frame, collageTop, hor_frame), axis=0)


if __name__ == '__main__':

    # Number of test images for each class !! REMINDER: Single_JPEG has label=1, Double_JPEG has label=0 !!
    N_TEST_IMG = 1500
    
    # Patch size
    IMG_SIZE = 256

    # ---------------------------------------------------------------------------------------------------------------------
    # DATA PREPARATION
    # ---------------------------------------------------------------------------------------------------------------------
    
    # Folders containing JPEG images (used to create the composite image with class activations)
    jpeg_path = r'D:\Datasets\jpeg_SQ_vs_DQ\256x256\aligned\60_95\test'
    single_jpeg_files_path = jpeg_path + '/single_pixel/*.jpg'
    double_jpeg_files_path = jpeg_path + '/double_pixel/*.jpg'

    # Get a file list, whose length depends on how many images we are considering
    single_files = sorted(glob(single_jpeg_files_path))
    double_files = sorted(glob(double_jpeg_files_path))
    jpeg_files = single_files[:N_TEST_IMG] + double_files[:N_TEST_IMG]

    # Load the test data (high-pass image component)
    hdf5_file = 'C:/Users/utente/PycharmProjects/TensorflowTests/data/singleVSdouble/output_data/jpeg_60_95_aligned_{}x{}.h5'.format(
        IMG_SIZE, IMG_SIZE)
    training_data, testing_data, training_labels, testing_labels = read_hdf5_compressed(hdf5_file)
 
    # ---------------------------------------------------------------------------------------------------------------------
    # MODEL PREPARATION
    # ---------------------------------------------------------------------------------------------------------------------
    
    # Load the trained CNN model
    modelpath = r'C:\Users\utente\PycharmProjects\TensorflowTests\single-vs-double_keras_{}x{}\trainedmodel\snapshots\final_trained_model.h5'.format(IMG_SIZE,IMG_SIZE)
    model = load_model(modelpath)

    # Check accuracy and then compute predictions and extrapolate class labels from them
    accuracy = model.predict(testing_data/255., verbose=1)
    print('Test accuracy {:g}. Loss {:g}'.format(accuracy[1], accuracy[0]))
 
    predicted_labels = np.argmax(predictions, axis=1)

    # Decode one-hot ground-truth labels stored in the input archive
    decoded_test_labels = np.argmax(testing_labels, axis=1)

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

    # Update frequency is controlled by this parameter
    ITER_VERBOSE = 250

    # Three different visualisation modes, the last one being the best
    modifier = 'guided'  # [None, 'relu', 'guided']

    # Classes: single JPEG (1) and double JPEG (0)
    classes = [0, 1]

    heatmap_single = np.zeros((N_TEST_IMG, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    heatmap_double = np.zeros((N_TEST_IMG, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    print()
    print('Computing localisation maps ... ')

    begin_time = time.time()
    n_files = testing_data.shape[0]
    for k in range(0, n_files):

        if k % ITER_VERBOSE == 0:
            print('Processed {:6d} of {:6d} images ({:3.1f}%. Elapsed: {:5.1f} seconds)'
                  .format(k, n_files, 100 * k / n_files, time.time() - begin_time))

        test_img = testing_data[k]

        # Localise only for the images that are correctly classified
        if decoded_test_labels[k] == predicted_labels[k]:

            # Tuple of size (modifiers, img_size, img_size, 3*number of classes), for example: (3, 64, 64, 6). In
            # practice, the i-th, j-th element (i, 64, 64, j) is the heatmap for the i-th modifier relatively
            # j-th class (j has 3 channels since it is a RGB map)
            heatmap_storage = np.zeros((1,) + test_img.shape[:2] + (3 * len(classes),), dtype=np.uint8)

            # Localisation of activating regions given a class cl = 1 (single)
            heatmap = visualize_cam(model,
                                    layer_idx,
                                    filter_indices=1,
                                    seed_input=test_img,
                                    backprop_modifier=modifier)

            # Localisation of activating regions given a class cl = 0 (double)
            heatmap_d = visualize_cam(model,
                                      layer_idx,
                                      filter_indices=0,
                                      seed_input=test_img,
                                      backprop_modifier=modifier)

            # Convert to RGB for display purposes
            if test_img.shape[2] < 3:
                overlayed_map = overlay(np.tile(test_img, (1, 1, 3)), heatmap)
                overlayed_map_d = overlay(np.tile(test_img, (1, 1, 3)), heatmap_d)

            # Store maps for single and double JPEG class
            heatmap_storage[0, :, :, 0:3] = overlayed_map
            heatmap_storage[0, :, :, 3:6] = overlayed_map_d

            # Create a composite image showing the original content, the high pass content and the heatmaps
            test_lowpass = scipy.misc.imread(jpeg_files[k])
            collage = make_collage(heatmap_storage, test_img, test_lowpass)

            # ------------------------------------------------------------------------------
            # Save the detection output as images
            # ------------------------------------------------------------------------------
            workdir = 'C:/Users/utente/PycharmProjects/FiltersVisualisation_Keras/Localisation'
            img_class_name = 'single' if decoded_test_labels[k] == 1 else 'double'
            img_name = os.path.basename(jpeg_files[k])
            scipy.misc.imsave('{}/{}/image_{}_{}_{}.png'.format(workdir,
                                                                img_class_name, k,
                                                                img_class_name,
                                                                os.path.splitext(img_name)[0]),
                              collage)

            # ------------------------------------------------------------------------------
            # Save activation maps as data matrices
            # ------------------------------------------------------------------------------
            np.savez(('{}/{}/matrixData_{}_{}_{}.npz'.format(workdir,
                                                             img_class_name, k,
                                                             img_class_name,
                                                             os.path.splitext(img_name)[0])),
                     gradcam_single=overlayed_map,
                     gradcam_double=overlayed_map_d)
