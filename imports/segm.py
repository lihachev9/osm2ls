import os
import json
from urllib.request import (
    pathname2url,
)  # for converting "+","*", etc. in file paths to appropriate urls
import logging

from imports.brush import image2annotation
from imports.utils import ExpandFullPath

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("root")
default_image_root_url = "/data/local-files/?d=images"


def convert_segm_to_ls(
    input_dir,
    out_file,
    to_name="image",
    from_name="tag",
    out_type="annotations",
    image_root_url=default_image_root_url,
    image_ext=".jpg,.jpeg,.png",
):
    tasks = []
    logger.info("Reading Segmentation notes and labels from %s", input_dir)

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
        task[out_type] = [
            {
                "result": [],
                "ground_truth": False,
            }
        ]
        label_dir = os.path.join(labels_dir, image_file_base)
        for i in os.listdir(label_dir):
            label_name = os.path.splitext(i)[0]
            annotation = image2annotation(
                path=label_dir + "/" + i,
                label_name=label_name,
                from_name=from_name,
                to_name=to_name
            )
            task[out_type][0]['result'].extend(annotation['result'])
        tasks.append(task)

    if len(tasks) > 0:
        logger.info("Saving Label Studio JSON to %s", out_file)
        with open(out_file, "w") as out:
            json.dump(tasks, out)
    else:
        logger.error("No labels converted")


def add_parser(subparsers):
    segm = subparsers.add_parser("segm")

    segm.add_argument(
        "-i",
        "--input",
        dest="input",
        required=True,
        help="directory with YOLO where images, labels, notes.json are located",
        action=ExpandFullPath,
    )
    segm.add_argument(
        "-o",
        "--output",
        dest="output",
        help="output file with Label Studio JSON tasks",
        default="output.json",
        action=ExpandFullPath,
    )
    segm.add_argument(
        "--to-name",
        dest="to_name",
        help="object name from Label Studio labeling config",
        default="image",
    )
    segm.add_argument(
        "--from-name",
        dest="from_name",
        help="control tag name from Label Studio labeling config",
        default="tag",
    )
    segm.add_argument(
        "--out-type",
        dest="out_type",
        help='annotation type - "annotations" or "predictions"',
        default="annotations",
    )
    segm.add_argument(
        "--image-root-url",
        dest="image_root_url",
        help="root URL path where images will be hosted, e.g.: http://example.com/images",
        default=default_image_root_url,
    )
    segm.add_argument(
        "--image-ext",
        dest="image_ext",
        help="image extension to search: .jpeg or .jpg, .png",
        default=".jpg,jpeg,.png",
    )
