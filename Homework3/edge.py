from tkinter import N
from turtle import Turtle
import numpy as np


def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp.

    Args:
        size: int of the size of output matrix.
        sigma: float of sigma to calculate kernel.

    Returns:
        kernel: numpy array of shape (size, size).
    """

    kernel = np.zeros((size, size))

    ### YOUR CODE HERE
    k = (size-1) / 2

    for i in range(size):
        x = np.exp(-(i - k) ** 2 / (2 * sigma ** 2))    # Reduce double counting
        for j in range(size):
            y = np.exp(-(j - k) ** 2 / (2 * sigma ** 2))
            kernel[i][j] = x * y / (2 * np.pi * sigma ** 2)
    ### END YOUR CODE

    ## 0.3s approximately faster than above
    # kernel = np.array([
    #     (np.exp(-(i - k) ** 2 / (2 * sigma ** 2)) / np.sqrt(2 * np.pi) / sigma) *
    #     (np.exp(-(j - k) ** 2 / (2 * sigma ** 2)) / np.sqrt(2 * np.pi) / sigma)
    #     for i in range(2 * k + 1)
    #         for j in range(2 * k + 1)
    # ]).reshape(size, size)

    return kernel


def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0, pad_width0), (pad_width1, pad_width1))
    padded = np.pad(image, pad_width, mode='edge')              # padded image
    
    ### YOUR CODE HERE
    kernel = np.flip(np.flip(kernel, 0), 1)

    for m in range(Hi):
        for n in range(Wi):
            out[m, n] = np.sum(padded[m:m+Hk, n:n+Wk] * kernel)
    ### END YOUR CODE
    return out


def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.
    """

    out = None
    ### YOUR CODE HERE
    kernel_x = np.array([[0.5, 0.0, -0.5]])
    out = conv(img, kernel_x)
    ### END YOUR CODE

    return out

def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    kernel_y = np.array([[0.5], [0.0], [-0.5]])
    out = conv(img, kernel_y)
    ### END YOUR CODE

    return out

def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use `np.sqrt` and `np.arctan2` to calculate square root and arctan
        - `np.arctan2` return angles in radians, you need to convert radians to degrees.
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE
    G = np.sqrt(partial_x(img)**2 + partial_y(img)**2)
    theta = np.rad2deg(np.arctan2(partial_y(img), partial_x(img)))
    theta = (theta + 180) % 360         # Some angles are negative, some are greater than 360.
    ### END YOUR CODE

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression.

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).

    Returns:
        out: non-maxima suppressed image.
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45
    theta = (theta % 360.0).astype(np.int32)

    #print(G)
    ### BEGIN YOUR CODE
    for i in range(H):
        for j in range(W):
            cur_angle = theta[i, j]
            # Compare the edge strength of the current pixel with the pixel in the positive and negative gradient directions. 
            if cur_angle == 0 or cur_angle == 180:
                if j == 0:
                    neighbors = [G[i, j+1]]
                elif j == W-1:
                    neighbors = [G[i, j-1]]
                else:
                    neighbors = [G[i, j-1], G[i, j+1]]
            elif cur_angle == 45 or cur_angle == 225:
                if j == 0 or i == 0:
                    neighbors = 0 if (i==H-1 or j==W-1) else [G[i+1, j+1]]
                elif j == W-1 or i == H-1:
                    neighbors = 0 if (i==0 or j==0) else [G[i-1, j-1]]
                else:
                    neighbors = [G[i-1, j-1], G[i+1, j+1]]
            elif cur_angle == 90 or cur_angle == 270:
                if i == 0:
                    neighbors = [G[i+1, j]]
                elif i == H-1:
                    neighbors = [G[i-1, j]]
                else:
                    neighbors = [G[i-1, j], G[i+1, j]]
            elif cur_angle == 135 or cur_angle == 315:
                if j == W-1 or i == 0:
                    neighbors = 0 if (i==H-1 or j==0) else [G[i+1, j-1]]
                elif j == 0 or i == H-1:
                    neighbors = 0 if (i==0 or j==W-1) else [G[i-1, j+1]]
                else:
                    neighbors = [G[i-1, j+1], G[i+1, j-1]]
            # If the edge strength of the current pixel is the largest, preserve. If not, suppress.
            if G[i,j] >= np.max(neighbors):
                out[i,j] = G[i,j]
            else:
                out[i, j] = 0
    ### END YOUR CODE

    return out

def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        high: high threshold(float) for strong edges.
        low: low threshold(float) for weak edges.

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edges are the pixels with the values greater than
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values smaller or equal to the
            higher threshold and greater than the lower threshold.
    """

    strong_edges = np.zeros(img.shape, dtype=np.bool)
    weak_edges = np.zeros(img.shape, dtype=np.bool)
    
    ### YOUR CODE HERE
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (img[i][j] > high):
                strong_edges[i][j] = True
            elif (img[i][j] > low):
                weak_edges[i][j]   = True
    ### END YOUR CODE

    # 01表示时，这样的写法更便捷。
    # strong_edges = np.zeros(img.shape)
    # weak_edges = np.zeros(img.shape)

    # strong_edges = img > high
    # weak_edges = (img > low) & (img < high)


    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x).

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel.
        H, W: size of the image.
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)].
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first search 
    across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W).
        weak_edges: binary image of shape (H, W).
    
    Returns:
        edges: numpy boolean array of shape(H, W).
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W), dtype=np.bool)

    # Make new instances of arguments to leave the original
    # references intact
    weak_edges = np.copy(weak_edges)
    edges = np.copy(strong_edges)

    ### YOUR CODE HERE
    for indice in indices:
        neigbours = get_neighbors(indice[0], indice[1], H, W)
        for neigbour in neigbours:
            if(weak_edges[neigbour] == True):       # Neighbors of strong edge pixels are also in weak edge pixels
                edges[neigbour] = True
    ### END YOUR CODE

    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W).
        kernel_size: int of size for kernel matrix.
        sigma: float for calculating kernel.
        high: high threshold for strong edges.
        low: low threashold for weak edges.
    Returns:
        edge: numpy array of shape(H, W).
    """
    # 1. Smoothing
    kernel = gaussian_kernel(kernel_size, sigma)
    smoothed = conv(img, kernel)
    # 2. Finding gradients
    G, theta = gradient(smoothed)
    # 3. Non-maximum suppression
    nms = non_maximum_suppression(G, theta)
    # 4. Double thresholding
    strong_edges, weak_edges = double_thresholding(nms, high, low)
    # 5. Edge tracking by hysterisis
    edge = link_edges(strong_edges, weak_edges)

    return edge