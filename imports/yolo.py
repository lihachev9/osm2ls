import os
import math
import json  # better to use "imports ujson as json" for the best performance
import uuid
import imagesize
from imports.label_config import generate_label_config
from imports.utils import defautl_parser, distance, new_task, default_image_root_url, logger


def convert_rectanglelabels(values, label, img_w, img_h):
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = \
        [(float(values[i]) * img_w, float(values[i + 1]) * img_h)
         for i in range(1, len(values), 2)]
    center_x = (x1 + x2 + x3 + x4) / 4
    center_y = (y1 + y2 + y3 + y4) / 4
    width = distance(x1, y1, x2, y2)
    height = distance(x1, y1, x4, y4)
    if width == 0 or height == 0:
        return
    dx = x2 - x1
    dy = y2 - y1
    rotation = math.degrees(math.atan2(dy, dx))

    # Find the top-left corner (x, y)
    radians = math.radians(rotation)
    cos, sin = math.cos(radians), math.sin(radians)
    height_2, width_2 = height / 2, width / 2
    top_left_x = center_x - width_2 * cos + height_2 * sin
    top_left_y = center_y - width_2 * sin - height_2 * cos

    x = (top_left_x / img_w) * 100
    y = (top_left_y / img_h) * 100
    width = (width / img_w) * 100
    height = (height / img_h) * 100

    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "rotation": rotation,
        "rectanglelabels": [label],
    }


def convert_polygonlabels(
    values, label, img_w=None, img_h=None
):
    points = [(float(values[i]) * 100, float(values[i + 1]) * 100)
              for i in range(1, len(values), 2)]

    return {"points": points, "polygonlabels": [label]}


def convert_yolo_to_ls(
    input_dir,
    out_file,
    to_name="image",
    from_name="label",
    out_type="annotations",
    image_root_url=default_image_root_url,
    image_ext=".jpg,.jpeg,.png",
    task_type="rectanglelabels",
):
    """Convert YOLO labeling to Label Studio JSON

    :param input_dir: directory with YOLO where images, labels, notes.json are located
    :param out_file: output file with Label Studio JSON tasks
    :param to_name: object name from Label Studio labeling config
    :param from_name: control tag name from Label Studio labeling config
    :param out_type: annotation type - "annotations" or "predictions"
    :param image_root_url: root URL path where images will be hosted, e.g.: http://example.com/images
    :param image_ext: image extension/s - single string or comma separated list to search, eg. .jpeg or .jpg, .png and so on.
    :param task_type: task type - "rectanglelabels" or "polygonlabels"
    """

    tasks = []
    task_fn = {
        "rectanglelabels": convert_rectanglelabels,
        "polygonlabels": convert_polygonlabels
    }[task_type]
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

    # generate and save labeling config
    label_config_file = out_file.replace('.json', '') + '.label_config.xml'
    generate_label_config(
        categories,
        {from_name: task_type},
        to_name,
        from_name,
        label_config_file,
    )

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
        task = new_task(
            out_type, image_root_url, image_file
        )

        # define coresponding label file and check existence
        label_file = os.path.join(labels_dir, image_file_base + ".txt")

        if os.path.exists(label_file):
            # read image sizes
            img_w, img_h = imagesize.get(os.path.join(images_dir, image_file))

            with open(label_file) as file:
                # convert all bounding boxes to Label Studio Results
                lines = file.readlines()
                for line in lines:
                    values = line.split()
                    label = categories[int(values[0])]
                    value = task_fn(
                        values, label, img_w, img_h
                    )
                    if value is None:
                        continue
                    item = {
                        "id": uuid.uuid4().hex[0:10],
                        "type": task_type,
                        "value": value,
                        "to_name": to_name,
                        "from_name": from_name,
                        "image_rotation": 0,
                        "original_width": img_w,
                        "original_height": img_h,
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
    parsers = defautl_parser(
        subparsers,
        "yolo",
        "directory with YOLO where images, labels, notes.json are located",
        "label"
    )
    parsers.add_argument(
        "--task_type",
        type=str,
        default="rectanglelabels",
        choices=["rectanglelabels", "polygonlabels"]
    )
