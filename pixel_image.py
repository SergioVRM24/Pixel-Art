import cv2
import numpy as np
from matplotlib import pyplot
import math

"""
1. Read image.
2. Binarize.
3. Cut.
4. Rize.
5. Rewrite using pixel styles.
6.
"""

def set_pixel_style(img):
    pixel = 255 + np.zeros((5,5))
    lx = img.shape[1]
    ly = img.shape[0]
    for y in range(ly):
        for x in range(lx):
            s = img[y, x]
            if x == 0:
                row = cv2.hconcat([np.zeros((5,1)), pixel*s, np.zeros((5,1))])
            else:
                row = cv2.hconcat([row, pixel*s, np.zeros((5,1))])
        if y == 0:
            pxl_img = cv2.vconcat([np.zeros((1,row.shape[1])), row, np.zeros((1,row.shape[1]))])
        else:
            pxl_img = cv2.vconcat([pxl_img, row, np.zeros((1,row.shape[1]))])
    return pxl_img

def scale_image(img, reduction_factor):
    w = round(img.shape[1]/reduction_factor)
    h = round(img.shape[0]/reduction_factor)
    sc_img = cv2.resize(img, (w, h))
    return sc_img

def autocrop_image(img):
    cx = np.sum(img, axis = 0)
    cy = np.sum(img, axis = 1)
    xi = np.where(cx > 0)
    yi = np.where(cy > 0)
    xi = [xi[0][0].astype(int), xi[0][-1].astype(int)]
    yi = [yi[0][0].astype(int), yi[0][-1].astype(int)]
    crop_img = img[yi[0]:yi[1],xi[0]:xi[1]]
    return crop_img

def binarize_image(img):
    _, bin_img = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY)
    return bin_img

def read_image(filename):
    img = cv2.imread(filename, 0)
    return img

def main():
    # Read image.
    img = read_image("rose.png")
    bin_img = binarize_image(img)
    # crop_img = autocrop_image(bin_img)
    sc_img = scale_image(bin_img, 40)
    pxl_img = set_pixel_style(sc_img)

    new_image = cv2.merge([pxl_img*0, pxl_img*0, pxl_img])

    cv2.imshow("Grayscale Image", new_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("image.png", new_image)

if __name__ == "__main__":
    main()