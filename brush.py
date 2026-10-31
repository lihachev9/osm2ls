import os
import cv2
import uuid
import rasterio
import numpy as np
import geopandas as gpd

from PIL import Image

from geojson_from_annotation import table_from_annotation
from utils import sanitize_filename


# Brush Export ###


class InputStream:
    def __init__(self, data):
        self.data = data
        self.i = 0

    def read(self, size):
        out = self.data[self.i: self.i + size]
        self.i += size
        return int(out, 2)


def access_bit(data, num):
    """from bytes array to bits by num position"""
    base = int(num // 8)
    shift = 7 - int(num % 8)
    return (data[base] & (1 << shift)) >> shift


def bytes2bit(data):
    """get bit string from bytes data"""
    return ''.join([str(access_bit(data, i)) for i in range(len(data) * 8)])


def decode_rle(rle, print_params: bool = False):
    """from LS RLE to numpy uint8 3d image [width, height, channel]

    Args:
        print_params (bool, optional): If true, a RLE parameters print statement is suppressed
    """
    input = InputStream(bytes2bit(rle))
    num = input.read(32)
    word_size = input.read(5) + 1
    rle_sizes = [input.read(4) + 1 for _ in range(4)]

    if print_params:
        print(
            'RLE params:', num, 'values', word_size, 'word_size', rle_sizes, 'rle_sizes'
        )

    i = 0
    out = np.zeros(num, dtype=np.uint8)
    while i < num:
        x = input.read(1)
        j = i + 1 + input.read(rle_sizes[input.read(2)])
        if x:
            val = input.read(word_size)
            out[i:j] = val
            i = j
        else:
            while i < j:
                val = input.read(word_size)
                out[i] = val
                i += 1
    return out


def decode_from_annotation(results):
    """from LS annotation to {"label_name": [numpy uint8 image (width x height)]}"""
    layers = {}
    for result in results:
        key = (
            'brushlabels'
            if result['type'].lower() == 'brushlabels'
            else ('labels' if result['type'].lower() == 'labels' else None)
        )
        if key is None:
            continue

        rle = result['value']['rle']
        width = result['original_width']
        height = result['original_height']
        labels = result['value'][key] if key in result['value'] else ['no_label']
        label = '-'.join(labels)

        image = np.reshape(decode_rle(rle), [height, width, 4])[:, :, 3]
        if label not in layers:
            layers[label] = image
        else:
            layers[label] = cv2.add(layers[label], image)
    return layers


def save_brush_images_from_annotation(
    results,
    out_dir: str,
    out_format: str,
    crs=None,
    transform=None
):
    if out_format == "tiff":
        if crs is None:
            raise Exception("crs is required for tiff output format")
        if transform is None:
            raise Exception("transformer is required for tiff output format")
    layers = decode_from_annotation(results)
    for label, layer in layers.items():
        label = sanitize_filename(label)
        filename = os.path.join(out_dir, label)
        if out_format == "numpy":
            np.save(filename, layer)
        elif out_format == "png":
            im = Image.fromarray(layer)
            im.save(filename + ".png")
        elif out_format == "tiff":
            with rasterio.open(
                filename + ".tiff", "w", driver="GTiff",
                height=layer.shape[0], width=layer.shape[1],
                count=1, dtype=layer.dtype, crs=crs, compress="deflate",
                transform=transform, nodata=0
            ) as dst:
                dst.write(layer, 1)
        else:
            raise Exception("Unknown output format for brush converter")


def save_geojson_from_annotation(results, filename, crs, transform):
    layers = decode_from_annotation(results)
    table = []
    labels = set()
    for label, layer in layers.items():
        label = sanitize_filename(label)
        table.extend(table_from_annotation(layer, label, transform))
        labels.add(label)
    gdf = gpd.GeoDataFrame(table, crs=crs)
    gdf.to_file(filename + ".geojson", driver="GeoJSON")
    # writeQML(labels, filename)


def convert_task(item, out_dir: str, out_format: str, crs=None, transform=None):
    """Task with multiple annotations to brush images, out_format = numpy | png | geojson"""
    for results in item:
        if out_format == "geojson":
            if crs is None:
                raise Exception("crs is required for geojson output format")
            if transform is None:
                raise Exception("transform is required for geojson output format")
            save_geojson_from_annotation(results, out_dir, crs, transform)
        else:
            save_brush_images_from_annotation(
                results, out_dir, out_format, crs, transform
            )


# Brush Import ###


def bits2byte(arr_str, n=8):
    """Convert bits back to byte

    :param arr_str:  string with the bit array
    :type arr_str: str
    :param n: number of bits to separate the arr string into
    :type n: int
    :return rle:
    :type rle: list
    """
    n_bytes = len(arr_str) // n
    return list(int(arr_str, 2).to_bytes(n_bytes, "big"))


def base_rle_encode(inarray):
    """run length encoding. Partial credit to R rle function.
    Multi datatype arrays catered for including non Numpy
    returns: tuple (runlengths, values)"""
    ia = np.asarray(inarray)  # force numpy
    n = len(ia)
    if n == 0:
        return None, None
    else:
        y = ia[1:] != ia[:-1]  # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)  # must include last element posi
        z = np.diff(np.append(-1, i))  # run lengths
        return z, ia[i]


def encode_rle(arr, wordsize=8, rle_sizes=(3, 4, 8, 16)):
    """Encode a 1d array to RLE (Label Studio / LSB-compatible)."""
    arr = np.asarray(arr).ravel()
    num = arr.size

    # ---- заголовок ----
    header_parts = [
        f"{num:032b}",                    # длина массива
        f"{wordsize - 1:05b}",            # wordsize
    ]
    header_parts.extend(f"{x - 1:04b}" for x in rle_sizes)
    header = "".join(header_parts)

    # ---- тело ----
    lengths, values = base_rle_encode(arr)

    # Быстрый путь для одиночных значений (самый частый случай)
    value_bits = [f"{int(v) & 0xFF:08b}" for v in values]

    chunks = []

    # Границы для rle_sizes = [3,4,8,16]
    # код длины: (2 бита индекса в rle_sizes) + (rle_sizes[idx] бит значения-1)
    # Одиночное значение кодируется как 0 + 00 + 000 + value
    max_per_size = [1 << s for s in rle_sizes]  # 8, 16, 256, 65536
    idx_bits = [f"{i:02b}" for i in range(len(rle_sizes))]

    for L, vb in zip(lengths.tolist(), value_bits):
        if L == 1:
            chunks.append("000000")
            chunks.append(vb)
            continue

        # series
        remaining = L
        # Для очень длинных серий режем на куски по 2**16
        big = max_per_size[-1]
        while remaining > big:
            chunks.append("1")
            chunks.append(idx_bits[-1])            # "11"
            chunks.append(f"{big - 1:016b}")
            chunks.append(vb)
            remaining -= big

        # Выбираем минимальный подходящий rle_size
        for idx, cap in enumerate(max_per_size):
            if remaining <= cap:
                chunks.append("1")
                chunks.append(idx_bits[idx])
                chunks.append(f"{remaining - 1:0{rle_sizes[idx]}b}")
                chunks.append(vb)
                break

    body = "".join(chunks)

    # ---- добивка до кратности 8 ----
    total = header + body
    pad = (-len(total)) % 8
    if pad:
        total += "0" * pad

    return bits2byte(total)


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
        rle = mask2rle(mask)
        return rle, image.size[0], image.size[1]


def create_result(
    rle,
    label_name,
    to_name,
    from_name,
    width,
    height
):
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


def mask2annotation(
    mask,
    label_name,
    from_name,
    to_name
):
    rle = mask2rle(mask)
    height, width = mask.shape
    return create_result(rle, label_name, to_name, from_name, width, height)


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
    return create_result(rle, label_name, to_name, from_name, width, height)
