import argparse
import os
import math
import logging

from urllib.request import (
    pathname2url,
)  # for converting "+","*", etc. in file paths to appropriate urls


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("root")
default_image_root_url = "/data/local-files/?d=images"


class ExpandFullPath(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))


def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def new_task(out_type, root_url, file_name):
    return {
        "data": {"image": os.path.join(root_url, pathname2url(file_name))},
        # 'annotations' or 'predictions'
        out_type: [
            {
                "result": [],
                "ground_truth": False,
            }
        ],
    }


def defautl_parser(subparsers, name, input_help, from_name):
    parsers = subparsers.add_parser(name)

    parsers.add_argument(
        "-i",
        "--input",
        dest="input",
        required=True,
        help=input_help,
        action=ExpandFullPath,
    )
    parsers.add_argument(
        "-o",
        "--output",
        dest="output",
        help="output file with Label Studio JSON tasks",
        default="output.json",
        action=ExpandFullPath,
    )
    parsers.add_argument(
        "--to-name",
        dest="to_name",
        help="object name from Label Studio labeling config",
        default="image",
    )
    parsers.add_argument(
        "--from-name",
        dest='from_name',
        help="control tag name from Label Studio labeling config",
        default=from_name,
    )
    parsers.add_argument(
        "--out-type",
        dest="out_type",
        help='annotation type - "annotations" or "predictions"',
        default="annotations",
    )
    parsers.add_argument(
        "--image-root-url",
        dest="image_root_url",
        help="root URL path where images will be hosted, e.g.: http://example.com/images",
        default=default_image_root_url,
    )
    parsers.add_argument(
        "--image-ext",
        dest="image_ext",
        help="image extension to search: .jpeg or .jpg, .png",
        default=".jpg,jpeg,.png",
    )

    return parsers
