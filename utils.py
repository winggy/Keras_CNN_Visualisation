"""
    File name: Visualise_filters.py
    Author: Andrea Costanzo, University of Siena, andreacos82@gmail.com
    Python Version: 3.5
"""

import os


def draw_captions(img_file, output_dir, labels = ['low-pass', 'high-pass', 'g-CAM single', 'g-CAM double']):

    """ Draws a text bar with labels on top of the image

    Args:
       img_file -- the path to the intput image
       output_dir -- the directory of the output image
    """

    from PIL import Image, ImageFont, ImageDraw

    source_img = Image.open(img_file).convert("RGBA")

    draw = ImageDraw.Draw(source_img)

    offset = 10
    for i in range(0, len(labels)):
        draw.text((offset,1), labels[i], font=ImageFont.truetype("arial", 35), fill="black")
        offset = offset + int((1/len(labels)) * source_img.width)

    # Get rid of the temp file
    os.remove(img_file)

    # Save new file
    source_img.save(output_dir, "PNG")

    return
