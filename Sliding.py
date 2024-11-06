def sliding_window(image, window_size, step_size):
    """
    This function generates sliding windows over an image.

    Parameters:
    image (ndarray): The input image over which the sliding window will be applied.
    window_size (tuple): The size of the window (width, height).
    step_size (tuple): The step size (horizontal step, vertical step) for moving the window.

    Yields:
    tuple: A tuple containing the x and y coordinates of the top-left corner of the window,
           and the window itself (a sub-region of the image).
    """
    for y in range(0, image.shape[0], step_size[1]):  # Loop over the image vertically with the given step size
        for x in range(0, image.shape[1], step_size[0]):  # Loop over the image horizontally with the given step size
            yield (x, y, image[y: y + window_size[1], x: x + window_size[0]]) # Extract the window from the image and yield the coordinates and the window