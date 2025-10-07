import os
import json

from brush import image2annotation
from imports.label_config import generate_label_config
from imports.utils import defautl_parser, new_task, default_image_root_url, logger


def convert_segm_to_ls(
    input_dir,
    out_file,
    to_name="image",
    from_name="tag",
    out_type="annotations",
    image_root_url=default_image_root_url,
    image_ext=".jpg,.jpeg,.png",
):
    """Convert Segmentation labeling to Label Studio JSON

    :param input_dir: directory with Segmentation where images, labels
    :param out_file: output file with Label Studio JSON tasks
    :param to_name: object name from Label Studio labeling config
    :param from_name: control tag name from Label Studio labeling config
    :param out_type: annotation type - "annotations" or "predictions"
    :param image_root_url: root URL path where images will be hosted, e.g.: http://example.com/images
    :param image_ext: image extension/s - single string or comma separated list to search, eg. .jpeg or .jpg, .png and so on
    """

    tasks = []
    logger.info("Reading Segmentation notes and labels from %s", input_dir)

    # define directories
    labels_dir = os.path.join(input_dir, "labels")
    images_dir = os.path.join(input_dir, "images")
    logger.info("Converting labels from %s", labels_dir)

    # build array out of provided comma separated image_extns (str -> array)
    image_ext = [x.strip() for x in image_ext.split(",")]
    logger.info(f"image extensions->, {image_ext}")
    categories = set()

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

        task = new_task(out_type, image_root_url, image_file)
        label_dir = os.path.join(labels_dir, image_file_base)
        for i in os.listdir(label_dir):
            label_name = os.path.splitext(i)[0]
            categories.add(label_name)
            annotation = image2annotation(
                path=label_dir + "/" + i,
                label_name=label_name,
                from_name=from_name,
                to_name=to_name
            )
            task[out_type][0]['result'].append(annotation)
        tasks.append(task)

    # generate and save labeling config
    categories = dict(enumerate(sorted(categories)))
    label_config_file = out_file.replace('.json', '') + '.label_config.xml'
    generate_label_config(
        categories,
        {from_name: "brushlabels"},
        to_name,
        from_name,
        label_config_file,
    )

    if len(tasks) > 0:
        logger.info("Saving Label Studio JSON to %s", out_file)
        with open(out_file, "w") as out:
            json.dump(tasks, out)
    else:
        logger.error("No labels converted")


def add_parser(subparsers):
    defautl_parser(
        subparsers,
        "segm",
        "input segmentation directory with images, labels",
        "tag"
    )
