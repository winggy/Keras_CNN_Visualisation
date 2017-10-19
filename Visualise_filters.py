"""
    File name: Visualise_filters.py
    Author: Andrea Costanzo, University of Siena, andreacos82@gmail.com
    Python Version: 3.5
"""

from __future__ import print_function

from scipy.misc import imsave
import numpy as np
import time
from keras.applications import vgg16
from keras import backend as K
from keras.models import load_model
import argparse

def deprocess_image(x):

    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def filter_visualisation(iterations=20, weights_path='imagenet',
                         layer='block5_conv1',  max_filters=200, filters=8, output_shape=224):

    """ Visualise convolutional filters of a CNN model by maximising their activations with gradient ascent.

    Args:
       iterations -- number of iterations of the gradient ascent procedure (Default: 20)
       weights_path -- path to the Keras CNN model (Default: imagenet model in Keras models directory)
       layer -- the network layer name whose activations are being shown. Use model.summary() to get names
       max_filters -- number of filters to be checked in layer
       filters -- determine the number of filters to be displayed (total: filters x filters)
       output_shape -- height and width of each output image showing activations (arranged in a grid)

    """

    img_width, img_height = output_shape, output_shape
    filter_indexes = range(0, max_filters)

    # By default, load the VGG16 network with ImageNet weights
    if weights_path == 'imagenet':
        model = vgg16.VGG16(weights=weights_path, include_top=False)
    else:
        model = load_model(weights_path)

    print('Model loaded.')

    # Display a summary of all the blocks
    model.summary()

    # This is the placeholder for the input images
    input_img = model.input

    # Get the symbolic outputs of each "key" layer (we gave them unique names)
    layer_dict = dict([(layer.name, layer) for layer in model.layers[0:]])

    kept_filters = []
    for filter_index in filter_indexes:

        # To speed up, scan only the first max_filters
        print('Processing filter %d' % filter_index)
        start_time = time.time()

        # Build a loss function that maximizes the activation of the nth filter of the layer considered
        layer_output = layer_dict[layer].output
        if K.image_data_format() == 'channels_first':
            loss = K.mean(layer_output[:, filter_index, :, :])
        else:
            loss = K.mean(layer_output[:, :, :, filter_index])

        # Compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        def normalize(x):
            # Utility function to normalize a tensor by its L2 norm
            return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

        # Normalization trick: we normalize the gradient
        grads = normalize(grads)

        # This function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # Step size for gradient ascent
        step = 1.

        # Start from a gray image with some random noise
        if K.image_data_format() == 'channels_first':
            input_img_data = np.random.random((1, 3, img_width, img_height))
        else:
            input_img_data = np.random.random((1, img_width, img_height, model.input.shape[-1]))
        input_img_data = (input_img_data - 0.5) * 20 + 128

        # Run gradient ascent for 20 steps
        for i in range(iterations):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step

            print('Current loss value:', loss_value)
            if loss_value <= 0.:
                # some filters get stuck to 0, we can skip them
                break

        # Decode the resulting input image
        if loss_value > 0:
            img = deprocess_image(input_img_data[0])
            kept_filters.append((img, loss_value))
        end_time = time.time()
        print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

    # Stich the best filters**2 filters on a filters x filters grid
    n = filters

    if filters**2 > len(kept_filters):
        n = int(np.floor(np.sqrt(len(kept_filters))))
    else:
        n = int(filters)

    # Filters that have the highest loss are assumed to be better-looking.
    kept_filters.sort(key=lambda x: x[1], reverse=True)
    kept_filters = kept_filters[:n * n]

    # Black picture with space for filters**2 filters with size output_shape x output_shape, 5px margin in between
    margin = 5
    width = n * img_width + (n - 1) * margin
    height = n * img_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))

    # Fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            img, loss = kept_filters[i * n + j]
            stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                             (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

    # Save the result to disk
    imsave('stitched_filters_%dx%d.png' % (n, n), stitched_filters)

    return stitched_filters


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20, help='Number of gradient ascent iterations')
    parser.add_argument("--weights", type=str, default='imagenet', help='Path to network weights file')
    parser.add_argument("--layer", type=str, default='block5_conv1', help='Name of layer to use.')
    parser.add_argument("--num_filters", type=int, default=8,
                        help='Number of filters to visualize, starting from filter number 0.')
    parser.add_argument("--max_filters", type=int, default=200,
                        help='Number of filters to visualize, starting from filter number 0.')
    parser.add_argument("--size", type=int, default=128, help='Image width and height')

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    print(args)

    """
    filter_visualisation(iterations=args.iterations,
                         weights_path=args.weights,
                         layer=args.layer,
                         filters=args.num_filters,
                         max_filters=args.max_filters,
                         output_shape=args.size)
                         """

    modelpath = r'C:\Users\apiccinnano\PycharmProjects\TensorflowTests\single-vs-double_keras\trainedmodel\snapshots\final_trained_model.h5'

    # First layer
    filter_visualisation(iterations=args.iterations,
                         weights_path=modelpath,
                         layer='conv1',
                         filters=6,
                         max_filters=19,
                         output_shape=64)

    # Second layer
    filter_visualisation(iterations=args.iterations,
                         weights_path=modelpath,
                         layer='conv2',
                         filters=6,
                         max_filters=49,
                         output_shape=64)
