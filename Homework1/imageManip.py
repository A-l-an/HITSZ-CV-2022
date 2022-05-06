import math

import numpy as np
from PIL import Image
from skimage import color, io


def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### YOUR CODE HERE
    # Use skimage io.imread
    out = io.imread(image_path)
    ### END YOUR CODE

    # Let's convert the image to be between the correct range.
    out = out.astype(np.float64) / 255
    return out


def crop_image(image, start_row, start_col, num_rows, num_cols):
    """Crop an image based on the specified bounds.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        start_row (int): The starting row index we want to include in our cropped image.
        start_col (int): The starting column index we want to include in our cropped image.
        num_rows (int): Number of rows in our desired cropped image.
        num_cols (int): Number of columns in our desired cropped image.

    Returns:
        out: numpy array of shape(num_rows, num_cols, 3).
    """

    out = None

    ### YOUR CODE HERE
    out = image[start_row:start_row+num_rows, start_col:start_col+num_cols, :]
    ### END YOUR CODE

    return out


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None

    ### YOUR CODE HERE
    # 如果写成out = image则会连带后面的resize的图片一起变暗
    out = np.zeros(shape=(image.shape[0], image.shape[1], 3)) 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pix = image[i:i+1, j:j+1, :]
            out[i:i+1, j:j+1, :] = 0.5*pix*pix
    ### END YOUR CODE

    return out


def resize_image(input_image, output_rows, output_cols):
    """Resize an image using the nearest neighbor method.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        output_rows (int): Number of rows in our desired output image.
        output_cols (int): Number of columns in our desired output image.

    Returns:
        np.ndarray: Resized image, with shape `(output_rows, output_cols, 3)`.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    # 1. Create the resized output image
    output_image = np.zeros(shape=(output_rows, output_cols, 3))

    # 2. Populate the `output_image` array using values from `input_image`
    #    > This should require two nested for loops!

    ### YOUR CODE HERE
    row_scale_factor = output_rows/input_rows   # 别计算错了
    col_scale_factor = output_cols/input_cols
    
    for output_i in range(output_rows):
        for output_j in range(output_cols):
            input_i = int(output_i/row_scale_factor)
            input_j = int(output_j/col_scale_factor)
            output_image[output_i:output_i+1, output_j:output_j+1, :] = input_image[input_i:input_i+1, input_j:input_j+1, :]

    ### END YOUR CODE

    # 3. Return the output image
    return output_image


def rotate2d(point, theta):
    """Rotate a 2D coordinate by some angle theta.

    Args:
        point (np.ndarray): A 1D NumPy array containing two values: an x and y coordinate.
        theta (float): An theta to rotate by, in radians.

    Returns:
        np.ndarray: A 1D NumPy array containing your rotated x and y values.
    """
    assert point.shape == (2,)
    assert isinstance(theta, float)

    # Reminder: np.cos() and np.sin() will be useful here!

    ## YOUR CODE HERE
    new_point = np.zeros(shape = point.shape)
    # new_point[0] = point[0]*np.cos(theta) - point[1]*np.sin(theta)
    # new_point[1] = point[1]*np.cos(theta) + point[0]*np.sin(theta)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    new_point = np.dot(R, point)
    return new_point
    ### END YOUR CODE


def rotate_image(input_image, theta):
    """Rotate an image by some angle theta.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        theta (float): Angle to rotate our image by, in radians.

    Returns:
        (np.ndarray): Rotated image, with the same shape as the input.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    # 1. Create an output image with the same shape as the input
    output_image = np.zeros_like(input_image)

    ## YOUR CODE HERE
    loc_x = input_image.shape[0]/2
    loc_y = input_image.shape[1]/2
    translate_lt = np.array([[1, 0, -loc_x], [0, 1, -loc_y], [0, 0, 1]])
    translate_rd = np.array([[1, 0, loc_x], [0, 1, loc_y], [0, 0, 1]])
    for i in range(input_image.shape[0]):
        for j in range(input_image.shape[1]):
            # pixel cannt be rotated, so caculate the coordinates
            loc_af_t = np.dot(translate_lt, np.array([i, j, 1]))   # 左上平移后的坐标
            loc_af_r = rotate2d(loc_af_t[0:2], -theta)             # 旋转后的坐标(rotate2d按逆时针来的，题目要求的是顺时针45)
            loc_af_t = np.dot(translate_rd, np.array([loc_af_r[0], loc_af_r[1], 1]))   # 右下平移后的坐标
            loc_o_x = int(loc_af_t[0])
            loc_o_y = int(loc_af_t[1])
            if (loc_o_x >= 0 and loc_o_x <= input_image.shape[0] and
                loc_o_y >= 0 and loc_o_y <= input_image.shape[1]):
                output_image[loc_o_x:loc_o_x+1, loc_o_y:loc_o_y+1, :] = input_image[i:i+1, j:j+1, :]   # 为了避免有的点因取整造成的丢失
                output_image[loc_o_x:loc_o_x+1, loc_o_y+1:loc_o_y+2, :] = input_image[i:i+1, j:j+1, :]
                output_image[loc_o_x+1:loc_o_x+2, loc_o_y:loc_o_y+1, :] = input_image[i:i+1, j:j+1, :]
                output_image[loc_o_x+1:loc_o_x+2, loc_o_y+1:loc_o_y+2, :] = input_image[i:i+1, j:j+1, :]
    ### END YOUR CODE

    # 3. Return the output image
    return output_image
