import argparse
import os
import math
import json  # better to use "imports ujson as json" for the best performance

import uuid
import logging

import imagesize
from typing import Optional, Tuple
from urllib.request import (
    pathname2url,
)  # for converting "+","*", etc. in file paths to appropriate urls


class ExpandFullPath(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))


logger = logging.getLogger("root")
default_image_root_url = "/data/local-files/?d=images"


def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def convert_yolo_to_ls(
    input_dir,
    out_file,
    to_name="image",
    from_name="label",
    out_type="annotations",
    image_root_url=default_image_root_url,
    image_ext=".jpg,.jpeg,.png",
    image_dims: Optional[Tuple[int, int]] = None,
):
    """Convert YOLO labeling to Label Studio JSON
    :param input_dir: directory with YOLO where images, labels, notes.json are located
    :param out_file: output file with Label Studio JSON tasks
    :param to_name: object name from Label Studio labeling config
    :param from_name: control tag name from Label Studio labeling config
    :param out_type: annotation type - "annotations" or "predictions"
    :param image_root_url: root URL path where images will be hosted, e.g.: http://example.com/images
    :param image_ext: image extension/s - single string or comma separated list to search, eg. .jpeg or .jpg, .png and so on.
    :param image_dims: image dimensions - optional tuple of integers specifying the image width and height of *all* images in the dataset. Defaults to opening the image to determine it's width and height, which is slower. This should only be used in the special case where you dataset has uniform image dimesions.
    """

    tasks = []
    logger.info("Reading YOLO notes and categories from %s", input_dir)

    # build categories=>labels dict
    notes_file = os.path.join(input_dir, "classes.txt")
    with open(notes_file) as f:
        lines = [line.strip() for line in f.readlines()]
    categories = {i: line for i, line in enumerate(lines)}
    logger.info(f"Found {len(categories)} categories")

    # define directories
    labels_dir = os.path.join(input_dir, "labels")
    images_dir = os.path.join(input_dir, "images")
    logger.info("Converting labels from %s", labels_dir)

    # build array out of provided comma separated image_extns (str -> array)
    image_ext = [x.strip() for x in image_ext.split(",")]
    logger.info(f"image extensions->, {image_ext}")

    # loop through images
    for f in os.listdir(images_dir):
        image_file_found_flag = False
        for ext in image_ext:
            if f.endswith(ext):
                image_file = f
                image_file_base = os.path.splitext(f)[0]
                image_file_found_flag = True
                break
        if not image_file_found_flag:
            continue

        image_root_url += "" if image_root_url.endswith("/") else "/"
        task = {
            "data": {
                # eg. '../../foo+you.py' -> '../../foo%2Byou.py'
                "image": image_root_url
                + str(pathname2url(image_file))
            }
        }

        # define coresponding label file and check existence
        label_file = os.path.join(labels_dir, image_file_base + ".txt")

        if os.path.exists(label_file):
            task[out_type] = [
                {
                    "result": [],
                    "ground_truth": False,
                }
            ]

            # read image sizes
            if image_dims is None:
                image_width, image_height = imagesize.get(os.path.join(images_dir, image_file))
            else:
                image_width, image_height = image_dims

            with open(label_file) as file:
                # convert all bounding boxes to Label Studio Results
                lines = file.readlines()
                for line in lines:
                    line_split = line.split()
                    label_id = line_split[0]
                    x1, y1, x2, y2, x3, y3, x4, y4 = [float(x) * 100 for x in line_split[1:]]
                    width = distance(x1, y1, x2, y2)
                    height = distance(x2, y2, x3, y3)
                    if width == 0 or height == 0:
                        continue
                    polygon = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    x, y = min(polygon, key=lambda pt: (pt[0],  pt[1]))
                    cos = (x2 - x1) / width
                    sin = (y2 - y1) / width
                    a_acos = math.acos(cos)
                    rotation = math.degrees(a_acos) if sin > 0 else math.degrees(-a_acos) % 360
                    item = {
                        "id": uuid.uuid4().hex[0:10],
                        "type": "rectanglelabels",
                        "value": {
                            "x": x,
                            "y": y,
                            "width": width,
                            "height": height,
                            "rotation": rotation,
                            "rectanglelabels": [categories[int(label_id)]],
                        },
                        "to_name": to_name,
                        "from_name": from_name,
                        "image_rotation": 0,
                        "original_width": image_width,
                        "original_height": image_height,
                    }
                    task[out_type][0]["result"].append(item)

        tasks.append(task)

    if len(tasks) > 0:
        logger.info("Saving Label Studio JSON to %s", out_file)
        with open(out_file, "w") as out:
            json.dump(tasks, out)
    else:
        logger.error("No labels converted")


def add_parser(subparsers):
    yolo = subparsers.add_parser("yolo")

    yolo.add_argument(
        "-i",
        "--input",
        dest="input",
        required=True,
        help="directory with YOLO where images, labels, notes.json are located",
        action=ExpandFullPath,
    )
    yolo.add_argument(
        "-o",
        "--output",
        dest="output",
        help="output file with Label Studio JSON tasks",
        default="output.json",
        action=ExpandFullPath,
    )
    yolo.add_argument(
        "--to-name",
        dest="to_name",
        help="object name from Label Studio labeling config",
        default="image",
    )
    yolo.add_argument(
        "--from-name",
        dest="from_name",
        help="control tag name from Label Studio labeling config",
        default="label",
    )
    yolo.add_argument(
        "--out-type",
        dest="out_type",
        help='annotation type - "annotations" or "predictions"',
        default="annotations",
    )
    yolo.add_argument(
        "--image-root-url",
        dest="image_root_url",
        help="root URL path where images will be hosted, e.g.: http://example.com/images",
        default=default_image_root_url,
    )
    yolo.add_argument(
        "--image-ext",
        dest="image_ext",
        help="image extension to search: .jpeg or .jpg, .png",
        default=".jpg,jpeg,.png",
    )
    yolo.add_argument(
        "--image-dims",
        dest="image_dims",
        type=int,
        nargs=2,
        help=(
            "optional tuple of integers specifying the image width and height of *all* "
            "images in the dataset. Defaults to opening the image to determine it's width "
            "and height, which is slower. This should only be used in the special "
            "case where you dataset has uniform image dimesions. e.g. `--image-dims 600 800` "
            "if all your images are of dimensions width=600, height=800"
        ),
        default=None,
    )


def get_all_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = False

    # Import
    parser_import = subparsers.add_parser(
        "import",
        help="Converter from external formats to Label Studio JSON annotations",
    )
    import_format = parser_import.add_subparsers(dest="import_format")
    add_parser(import_format)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_all_args()
    convert_yolo_to_ls(
        input_dir=args.input,
        out_file=args.output,
        to_name=args.to_name,
        from_name=args.from_name,
        out_type=args.out_type,
        image_root_url=args.image_root_url
    )
