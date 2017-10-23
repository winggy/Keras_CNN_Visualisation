import os


def draw_captions(img_file, output_dir=None):

    """ Draws a text bar with labels on top of the image

    Args:
       img_file -- the path to the intput image
       output_dir -- the directory of the output image
    """

    from PIL import Image, ImageFont, ImageDraw


    source_img = Image.open(img_file).convert("RGBA")

    draw = ImageDraw.Draw(source_img)

    labels = ['content', 'high-pass', 'g-CAM single', 'g-CAM double']
    offset = 10
    for i in range(0,4):
        draw.text((offset,1), labels[i], font=ImageFont.truetype("arial", 35), fill="red")
        offset = offset + int(0.25 * source_img.width)

    # Get rid of the temp file
    os.remove(img_file)

    # Save new file
    if output_dir is None:
        source_img.save('final_map.png', "PNG")
    else:
        source_img.save(os.path.join(output_dir, 'final_map.png'), "PNG")

    return
