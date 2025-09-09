import io
import logging
import os
from enum import Enum
from collections import defaultdict
from copy import deepcopy

import ijson
import ujson as json
from utils import get_json_root_type
from brush import convert_task_dir


logger = logging.getLogger(__name__)


class Format(Enum):
    JSON = 1
    JSON_MIN = 2
    CSV = 3
    TSV = 4
    CONLL2003 = 5
    COCO = 6
    VOC = 7
    BRUSH_TO_NUMPY = 8
    BRUSH_TO_PNG = 9
    ASR_MANIFEST = 10
    YOLO = 11
    YOLO_OBB = 12
    CSV_OLD = 13
    YOLO_WITH_IMAGES = 14
    COCO_WITH_IMAGES = 15
    YOLO_OBB_WITH_IMAGES = 16
    BRUSH_TO_COCO = 17

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s):
        try:
            return Format[s]
        except KeyError:
            raise ValueError()


class Converter():
    _FORMAT_INFO = {
        Format.BRUSH_TO_NUMPY: {
            "title": "Brush labels to NumPy",
            "description": "Export your brush labels as NumPy 2d arrays. Each label outputs as one image.",
            "tags": ["image segmentation"],
        },
        Format.BRUSH_TO_PNG: {
            "title": "Brush labels to PNG",
            "description": "Export your brush labels as PNG images. Each label outputs as one image.",
            "tags": ["image segmentation"],
        },
    }

    def convert(self, input_data, output_data, format):
        if isinstance(format, str):
            format = Format.from_string(format)

        if format == Format.BRUSH_TO_NUMPY:
            items = self.iter_from_json_file(input_data)
            basename = os.path.splitext(os.path.basename(input_data))[0]
            output_data = os.path.join(output_data, basename)
            convert_task_dir(items, output_data, out_format="numpy")

    def iter_from_json_file(self, json_file):
        """Extract annotation results from json file

        param json_file: path to task list or dict with annotations
        """
        data_type = get_json_root_type(json_file)

        # one task
        if data_type == "dict":
            with open(json_file, "r") as json_file:
                data = json.load(json_file)
            for item in self.annotation_result_from_task(data):
                yield item

        # many tasks
        elif data_type == "list":
            with io.open(json_file, "rb") as f:
                data = ijson.items(
                    f, "item", use_float=True
                )  # 'item' means to read array of dicts
                for task in data:
                    for item in self.annotation_result_from_task(task):
                        if item is not None:
                            yield item

    def annotation_result_from_task(self, task):
        has_annotations = "completions" in task or "annotations" in task
        if not has_annotations:
            logger.warning(
                'Each task dict item should contain "annotations" or "completions" [deprecated], '
                "where value is list of dicts"
            )
            return None

        # get last not skipped completion and make result from it
        annotations = (
            task["annotations"] if "annotations" in task else task["completions"]
        )

        # return task with empty annotations
        if not annotations:
            data = Converter.get_data(task, {}, {})
            yield data

        # skip cancelled annotations
        cancelled = lambda x: not (
            x.get("skipped", False) or x.get("was_cancelled", False)
        )
        annotations = list(filter(cancelled, annotations))
        if not annotations:
            return None

        # sort by creation time
        annotations = sorted(
            annotations, key=lambda x: x.get("created_at", 0), reverse=True
        )

        for annotation in annotations:
            result = annotation["result"]
            outputs = defaultdict(list)

            # get results only as output
            for r in result:
                if "from_name" in r and (
                    tag_name := self._maybe_matching_tag_from_schema(r["from_name"])
                ):
                    v = deepcopy(r["value"])
                    v["type"] = self._schema[tag_name]["type"]
                    if "original_width" in r:
                        v["original_width"] = r["original_width"]
                    if "original_height" in r:
                        v["original_height"] = r["original_height"]
                    outputs[r["from_name"]].append(v)

            data = Converter.get_data(task, outputs)
            if "agreement" in task:
                data["agreement"] = task["agreement"]
            yield data

    @staticmethod
    def get_data(task, outputs):
        return {
            "id": task["id"],
            "input": task["data"],
            "output": outputs or {}
        }
