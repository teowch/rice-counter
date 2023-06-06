# ===============================================================================
# Author: Teodoro Valença de Souza Wacholski
# Universidade Tecnológica Federal do Paraná
# ===============================================================================

import sys
import timeit
import numpy as np
import cv2

INPUT_IMAGE = "rice.bmp"

# You can adjust these params
NEGATIVE = False
THRESHOLD = 0.8
MIN_HEIGHT = 10
MIN_WIDTH = 10
MIN_PIXELS = 20


def binarize(img, threshold):
    """
    Binarization by thresholding
    Params:
        img: input image
        threshold: threshold
            Every value < threshold will be set to 0
            Otherwise will be set to 1
    Returns:
        A binarized version of img
    """

    rows, cols, channels = img.shape
    for row in range(rows):
        for col in range(cols):
            if img[row, col] < threshold:
                img[row, col] = 0
            else:
                img[row, col] = 1
    return img


def flood(label, labelMatrix, y0, x0, n_pixels):
    """
    Flood fill algorithm to label connected components.
    Params:
        label: label value to assign to the connected component
        labelMatrix: matrix representing the labels of each pixel of the image
        y0: starting y-coordinate for flood fill
        x0: starting x-coordinate for flood fill
        n_pixels: current number of pixels in the connected component
    Returns:
        A dictionary containing information about the connected component:
            'T': top edge of the component
            'L': left edge of the component
            'B': bottom edge of the component
            'R': right edge of the component
            'n_pixels': number of pixels in the connected component
    """

    labelMatrix[y0, x0] = label
    rows, cols = labelMatrix.shape

    n_pixels += 1
    n = 0

    # Temporary storage of flood output to compare to info
    temp = {"T": y0, "L": x0, "B": y0, "R": x0, "n_pixels": 0}

    # Flood function output
    info = {
        "T": temp["T"],
        "L": temp["L"],
        "B": temp["B"],
        "R": temp["R"],
        "n_pixels": n_pixels + n,
    }

    # Neighbors array to iterate, taking care to image bounds
    neighbors = [
        labelMatrix[y0 + 1, x0] if (y0 + 1) < rows else 0,
        labelMatrix[y0, x0 + 1] if (x0 + 1) < cols else 0,
        labelMatrix[y0, x0 - 1] if (x0 - 1) >= 0 else 0,
        labelMatrix[y0 - 1, x0] if (y0 - 1) >= 0 else 0,
    ]
    neighborsIndex = [[y0 + 1, x0], [y0, x0 + 1], [y0, x0 - 1], [y0 - 1, x0]]

    # For each neighbor...
    for index in range(len(neighbors)):
        # Check for image bounds
        if (
            (index == 0 and (y0 + 1) < rows)
            or (index == 1 and (x0 + 1) < cols)
            or (index == 2 and (x0 - 1) >= 0)
            or (index == 3 and (y0 - 1) >= 0)
        ):
            # If the neighbor is a pixel of interest and was not visited
            if neighbors[index] == -1:
                # Flood fill in the neighbor
                temp = flood(
                    label,
                    labelMatrix,
                    neighborsIndex[index][0],
                    neighborsIndex[index][1],
                    n_pixels,
                )
        # Verify if the bounds have increased
        if temp["T"] < info["T"]:
            info["T"] = temp["T"]
        if temp["B"] > info["B"]:
            info["B"] = temp["B"]
        if temp["L"] < info["L"]:
            info["L"] = temp["L"]
        if temp["R"] > info["R"]:
            info["R"] = temp["R"]

        # Sum temp pixels to flood output
        n += temp["n_pixels"]

    info["n_pixels"] = n_pixels + n

    return info


def labeling(img, min_width, min_height, min_pixels):
    """
    Labeling using Flood Fill.
    Params:
        img: input image
        min_width: Discard components with width < min_width
        min_height: Discard components with height < min_height
        min_pixels: Discard components with less than min_pixels
    Returns:
        A list, where each item is a dictionary with the fields:
            'label': component label
            'n_pixels': component number of pixels
            'T', 'L', 'B', 'R': component edge coordinates (Top, Left, Bottom and Right)
    """

    rows, cols, channels = img.shape

    labelMatrix = np.empty((rows, cols))
    outputList = []

    # Labels the pixels of interest as unvisited
    for row in range(rows):
        for col in range(cols):
            if img[row, col] != 0:
                labelMatrix[row, col] = -1

    sys.setrecursionlimit(5000)
    label = 1

    # For each pixel...
    for row in range(rows):
        for col in range(cols):
            if labelMatrix[row, col] == -1:
                n_pixels = 0
                info = flood(label, labelMatrix, row, col, n_pixels)
                component = {
                    "label": label,
                    "n_pixels": info["n_pixels"],
                    "T": info["T"],
                    "L": info["L"],
                    "B": info["B"],
                    "R": info["R"],
                }
                if component["n_pixels"] > min_pixels:
                    if (component["B"] - component["T"] > min_height) and (
                        component["R"] - component["L"] > min_width
                    ):
                        outputList.append(component)
                        label += 1

    return outputList


def main():
    img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Cannot open the image.\n")
        sys.exit()

    # Preprocess the image
    img = img.reshape((img.shape[0], img.shape[1], 1))
    img = img.astype(np.float32) / 255
    img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Binarize the image
    if NEGATIVE:
        img = 1 - img
    img = binarize(img, THRESHOLD)

    cv2.imshow("00 - original", img_out)
    cv2.imshow("01 - binarized", img)
    cv2.imwrite("01 - binarized.png", img * 255)

    start_time = timeit.default_timer()

    # Perform labeling using flood fill algorithm
    components = labeling(img, MIN_WIDTH, MIN_HEIGHT, MIN_PIXELS)
    n_components = len(components)
    print("Time: %.2fs" % (timeit.default_timer() - start_time))
    print("%d components detected." % n_components)

    # Draw rectangles around the labeled components
    for c in components:
        cv2.rectangle(img_out, (c["L"], c["T"]), (c["R"], c["B"]), (0, 0, 1))

    cv2.imshow("02 - out", img_out)
    cv2.imwrite("02 - out.png", img_out * 255)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
