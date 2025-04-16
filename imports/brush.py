import uuid
import numpy as np

from PIL import Image


def bits2byte(arr_str, n=8):
    """Convert bits back to byte

    :param arr_str:  string with the bit array
    :type arr_str: str
    :param n: number of bits to separate the arr string into
    :type n: int
    :return rle:
    :type rle: list
    """
    rle = []
    numbers = [arr_str[i : i + n] for i in range(0, len(arr_str), n)]
    for i in numbers:
        rle.append(int(i, 2))
    return rle


# Shamelessly plagiarized from https://stackoverflow.com/a/32681075/6051733
def base_rle_encode(inarray):
    """run length encoding. Partial credit to R rle function.
    Multi datatype arrays catered for including non Numpy
    returns: tuple (runlengths, startpositions, values)"""
    ia = np.asarray(inarray)  # force numpy
    n = len(ia)
    if n == 0:
        return None, None, None
    else:
        y = ia[1:] != ia[:-1]  # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)  # must include last element posi
        z = np.diff(np.append(-1, i))  # run lengths
        p = np.cumsum(np.append(0, z))[:-1]  # positions
        return z, p, ia[i]


def encode_rle(arr, wordsize=8, rle_sizes=[3, 4, 8, 16]):
    """Encode a 1d array to rle


    :param arr: flattened np.array from a 4d image (R, G, B, alpha)
    :type arr: np.array
    :param wordsize: wordsize bits for decoding, default is 8
    :type wordsize: int
    :param rle_sizes:  list of ints which state how long a series is of the same number
    :type rle_sizes: list
    :return rle: run length encoded array
    :type rle: list

    """
    # Set length of array in 32 bits
    num = len(arr)
    numbits = f"{num:032b}"

    # put in the wordsize in bits
    wordsizebits = f"{wordsize - 1:05b}"

    # put rle sizes in the bits
    rle_bits = "".join([f"{x - 1:04b}" for x in rle_sizes])

    # combine it into base string
    base_str = numbits + wordsizebits + rle_bits

    # start with creating the rle bite string
    out_str = ""
    for length_reeks, p, value in zip(*base_rle_encode(arr)):
        # TODO: A nice to have but --> this can be optimized but works
        if length_reeks == 1:
            # we state with the first 0 that it has a length of 1
            out_str += "0"
            # We state now the index on the rle sizes
            out_str += "00"

            # the rle size value is 0 for an individual number
            out_str += "000"

            # put the value in a 8 bit string
            out_str += f"{value:08b}"

        elif length_reeks > 1:
            # rle size = 3
            if length_reeks <= 8:
                # Starting with a 1 indicates that we have started a series
                out_str += "1"

                # index in rle size arr
                out_str += "00"

                # length of array to bits
                out_str += f"{length_reeks - 1:03b}"

                out_str += f"{value:08b}"

            # rle size = 4
            elif 8 < length_reeks <= 16:
                # Starting with a 1 indicates that we have started a series
                out_str += "1"
                out_str += "01"

                # length of array to bits
                out_str += f"{length_reeks - 1:04b}"

                out_str += f"{value:08b}"

            # rle size = 8
            elif 16 < length_reeks <= 256:
                # Starting with a 1 indicates that we have started a series
                out_str += "1"

                out_str += "10"

                # length of array to bits
                out_str += f"{length_reeks - 1:08b}"

                out_str += f"{value:08b}"

            # rle size = 16 or longer
            else:
                length_temp = length_reeks
                while length_temp > 2**16:
                    # Starting with a 1 indicates that we have started a series
                    out_str += "1"

                    out_str += "11"
                    out_str += f"{2 ** 16 - 1:016b}"

                    out_str += f"{value:08b}"
                    length_temp -= 2**16

                # Starting with a 1 indicates that we have started a series
                out_str += "1"

                out_str += "11"
                # length of array to bits
                out_str += f"{length_temp - 1:016b}"

                out_str += f"{value:08b}"

    # make sure that we have an 8 fold lenght otherwise add 0's at the end
    nzfill = 8 - len(base_str + out_str) % 8
    total_str = base_str + out_str
    total_str = total_str + nzfill * "0"

    rle = bits2byte(total_str)

    return rle


def contour2rle(contours, contour_id, img_width, img_height):
    """
    :param contours:  list of contours
    :type contours: list
    :param contour_id: id of contour which you want to translate
    :type contour_id: int
    :param img_width: image shape width
    :type img_width: int
    :param img_height: image shape height
    :type img_height: int
    :return: list of ints in RLE format
    """
    import cv2  # opencv

    mask_im = np.zeros((img_width, img_height, 4))
    mask_contours = cv2.drawContours(
        mask_im, contours, contour_id, color=(0, 255, 0, 100), thickness=-1
    )
    rle_out = encode_rle(mask_contours.ravel().astype(int))
    return rle_out


def mask2rle(mask):
    """Convert mask to RLE

    :param mask: uint8 or int np.array mask with len(shape) == 2 like grayscale image
    :return: list of ints in RLE format
    """
    assert len(mask.shape) == 2, "mask must be 2D np.array"
    assert mask.dtype == np.uint8 or mask.dtype == int, "mask must be uint8 or int"
    array = mask.ravel()
    array = np.repeat(array, 4)  # must be 4 channels
    rle = encode_rle(array)
    return rle


def image2rle(path):
    """Convert mask image (jpg, png) to RLE

    1. Read image as grayscale
    2. Flatten to 1d array
    3. Threshold > 128
    4. Encode

    :param path: path to image with mask (jpg, png), this image will be thresholded with values > 128 to obtain mask,
                 so you can mark background as black and foreground as white
    :return: list of ints in RLE format
    """
    with Image.open(path).convert("L") as image:
        mask = np.array((np.array(image) > 128) * 255, dtype=np.uint8)
        array = mask.ravel()
        array = np.repeat(array, 4)
        rle = encode_rle(array)
        return rle, image.size[0], image.size[1]


def image2annotation(
    path,
    label_name,
    from_name,
    to_name
):
    """Convert image with mask to brush RLE annotation

    :param path: path to image with mask (jpg, png), this image will be thresholded with values > 128 to obtain mask,
                 so you can mark background as black and foreground as white
    :param label_name: label name from labeling config (<Label>)
    :param from_name: brush tag name (<BrushLabels>)
    :param to_name: image tag name (<Image>)

    :return: dict with Label Studio Annotation or Prediction (Pre-annotation)
    """
    rle, width, height = image2rle(path)
    result = {
        "id": str(uuid.uuid4())[0:8],
        "type": "brushlabels",
        "value": {"rle": rle, "format": "rle", "brushlabels": [label_name]},
        "origin": "manual",
        "to_name": to_name,
        "from_name": from_name,
        "image_rotation": 0,
        "original_width": width,
        "original_height": height,
    }
    return result
