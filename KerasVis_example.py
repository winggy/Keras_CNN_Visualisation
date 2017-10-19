
import os
from vis.visualization import visualize_saliency, overlay, visualize_cam, visualize_activation
from vis.input_modifiers import Jitter
from vis.utils import utils
from matplotlib import pyplot as plt
from keras import activations
from keras.applications import VGG16
from keras.models import load_model

# Build the VGG16 network with ImageNet weights
model = VGG16(weights='imagenet', include_top=True)

# Utility to search for layer index by name. Alternatively, set this to -1 since it corresponds to the last layer
layer_idx = utils.find_layer_idx(model, 'predictions')

# Swap SoftMax with linear: it provides better visualisation results
model.layers[layer_idx].activation = activations.linear

# To update model, we must save it and reload it
model.save('tmp.h5')
model = load_model('tmp.h5')
os.remove('tmp.h5')

plt.rcParams['figure.figsize'] = (18, 6)

img1 = utils.load_img('images/ouzel1.jpg', target_size=(224, 224))
img2 = utils.load_img('images/ouzel2.jpg', target_size=(224, 224))

# 20 is the imagenet category for 'ouzel'
# img = visualize_activation(model, layer_idx, filter_indices=20)

img = visualize_activation(model, layer_idx, filter_indices=20, max_iter=500, input_modifiers=[Jitter(16)])

plt.imshow(img)
plt.show()

for modifier in [None, 'guided', 'relu']:
    f, ax = plt.subplots(1, 2)
    plt.suptitle("vanilla" if modifier is None else modifier)
    for i, img in enumerate([img1, img2]):
        # 20 is the imagenet index corresponding to `ouzel`
        heatmap = visualize_cam(model, layer_idx, filter_indices=20,
                                seed_input=img, backprop_modifier=modifier)
        # Lets overlay the heatmap onto original image.
        ax[i].imshow(overlay(img, heatmap))

    f.show()
