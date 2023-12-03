import argparse
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

SPI = 14.0 # stitches per inch
DIAMETER = 5.0 # inches

def dimensions(img: Image.Image):
    img_h_w = np.array([img.height, img.width], dtype=np.float64)
    pixel_max = max(img_h_w)
    img_h_w = (img_h_w / pixel_max) * DIAMETER * SPI
    img_h_w = img_h_w.round()
    return img_h_w.astype(np.uint32)

# TODO do a partial weighting if the pixel is not fully included
# TODO needs optimization
def majority_resize(img: Image.Image, height: int, width: int):
    pixel_h = img.height / height
    pixel_w = img.width / width

    colors = [[defaultdict(int) for w in range(width)] for h in range(height)]
    img_data = np.asarray(img)
    for row_i, row in enumerate(img_data):
        for col_i, pix in enumerate(row):
            colors[int(row_i//pixel_h)][int(col_i//pixel_w)][int.from_bytes(bytearray(pix), byteorder='big', signed=False)] += 1.0
    
    result = []
    for row in colors:
        result_row = []
        for col in row:
            most_common = max(col, key=col.get)
            result_row.append(list(most_common.to_bytes(4, byteorder='big', signed=False)))
        result.append(result_row)
    
    return np.array(result, dtype=np.uint8)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TODO')
    parser.add_argument('image_file', type=str, help='path to the image file to be processed')
    args = parser.parse_args()

    filepath = os.path.abspath(args.image_file)

    img = Image.open(filepath)
    img_dim = dimensions(img)
    # img = img.resize(img_dim[::-1], Image.Resampling.BOX)
    img_data = majority_resize(img, img_dim[0], img_dim[1])
    # img_data = np.asarray(img)

    # img.show()
    
    # img_data = np.asarray(img)
    Image.fromarray(img_data).save(os.path.splitext(filepath)[0] + f'_{img_dim}' + '.png')
    plt.imshow(img_data, interpolation='nearest')
    plt.show()

    pass