import json  # better to use "imports ujson as json" for the best performance
import uuid

from imports.utils import defautl_parser, new_task, logger
from imports.label_config import generate_label_config


def create_segmentation(
    category_id, segmentation, categories, from_name, image_height, image_width, to_name
):
    label = categories[int(category_id)]
    points = [list(x) for x in zip(*[iter(segmentation)] * 2)]

    for i in range(len(points)):
        points[i][0] = points[i][0] / image_width * 100.0
        points[i][1] = points[i][1] / image_height * 100.0

    item = {
        "id": uuid.uuid4().hex[0:10],
        "type": "polygonlabels",
        "value": {"points": points, "polygonlabels": [label]},
        "to_name": to_name,
        "from_name": from_name,
        "image_rotation": 0,
        "original_width": image_width,
        "original_height": image_height,
    }
    return item


def convert_coco_to_ls(
    input_file,
    out_file,
    to_name="image",
    from_name="label",
    out_type="annotations",
    image_root_url="/data/local-files/?d=",
    use_super_categories=False
):
    """Convert COCO labeling to Label Studio JSON

    :param input_file: file with COCO json
    :param out_file: output file with Label Studio JSON tasks
    :param to_name: object name from Label Studio labeling config
    :param from_name: control tag name from Label Studio labeling config
    :param out_type: annotation type - "annotations" or "predictions"
    :param image_root_url: root URL path where images will be hosted, e.g.: http://example.com/images
    :param use_super_categories: use super categories from categories if they are presented
    :param point_width: key point width
    """

    tasks = {}  # image_id => task
    logger.info("Reading COCO notes and categories from %s", input_file)

    with open(input_file, encoding="utf8") as f:
        coco = json.load(f)

    # build categories => labels dict
    new_categories = {}
    # list to dict conversion: [...] => {category_id: category_item}
    categories = {int(category["id"]): category for category in coco["categories"]}
    ids = sorted(categories.keys())  # sort labels by their origin ids

    for i in ids:
        name = categories[i]["name"]
        if use_super_categories and "supercategory" in categories[i]:
            name = categories[i]["supercategory"] + ":" + name
        new_categories[i] = name

    # mapping: id => category name
    categories = new_categories

    # mapping: image id => image
    images = {item["id"]: item for item in coco["images"]}

    logger.info(
        f'Found {len(categories)} categories, {len(images)} images and {len(coco["annotations"])} annotations'
    )

    # flags for labeling config composing
    segmentation = bbox = keypoints = rle = False
    segmentation_once = bbox_once = keypoints_once = rle_once = False
    segmentation_from_name = from_name + "polygons"
    tags = {}

    # create tasks
    for image in coco["images"]:
        image_id, image_file_name = image["id"], image["file_name"]
        tasks[image_id] = new_task(out_type, image_root_url, image_file_name)

    for i, annotation in enumerate(coco["annotations"]):
        segmentation |= "segmentation" in annotation
        bbox |= "bbox" in annotation
        keypoints |= "keypoints" in annotation
        rle |= (
            annotation.get("iscrowd") == 1
        )  # 0 - polygons are in segmentation, otherwise rle

        if rle and not rle_once:  # not supported
            logger.error("RLE in segmentation is not yet supported")
            rle_once = True
        if keypoints and not keypoints_once:
            logger.error("Keypoints is not yet supported")
            keypoints_once = True
        if segmentation and not segmentation_once:  # not supported
            logger.warning("Segmentation in COCO is experimental")
            tags.update({segmentation_from_name: "PolygonLabels"})
            segmentation_once = True
        if bbox and not bbox_once:
            logger.error("Bbox is not yet supported")
            bbox_once = True

        # read image sizes
        image_id = annotation["image_id"]
        image = images[image_id]
        image_file_name, image_width, image_height = (
            image["file_name"],
            image["width"],
            image["height"],
        )

        task = tasks[image_id]

        if "segmentation" in annotation and len(annotation["segmentation"]):
            for single_segmentation in annotation["segmentation"]:
                item = create_segmentation(
                    annotation["category_id"],
                    single_segmentation,
                    categories,
                    segmentation_from_name,
                    image_height,
                    image_width,
                    to_name,
                )
                task[out_type][0]["result"].append(item)

        tasks[image_id] = task

    # generate and save labeling config
    label_config_file = out_file.replace(".json", "") + ".label_config.xml"
    generate_label_config(categories, tags, to_name, from_name, label_config_file)

    if len(tasks) > 0:
        tasks = [tasks[key] for key in sorted(tasks.keys())]
        logger.info("Saving Label Studio JSON to %s", out_file)
        with open(out_file, "w") as out:
            json.dump(tasks, out)

        print(
            "\n"
            f"  1. Create a new project in Label Studio\n"
            f'  2. Use Labeling Config from "{label_config_file}"\n'
            f"  3. Setup serving for images [e.g. you can use Local Storage (or others):\n"
            f"     https://labelstud.io/guide/storage.html#Local-storage]\n"
            f'  4. Import "{out_file}" to the project\n'
        )
    else:
        logger.error("No labels converted")


def add_parser(subparsers):
    defautl_parser(
        subparsers,
        "coco",
        "input COCO json file",
        "label"
    )
