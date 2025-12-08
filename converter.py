import logging
import os

import ijson
import ujson as json
from brush import convert_task
from utils import get_json_root_type
from imports.utils import ExpandFullPath


logger = logging.getLogger(__name__)


class Converter():
    def convert(self, input_data: str, output_data: str, format: str):
        if os.path.isdir(input_data):
            for json_file in self.iter_from_dir(input_data):
                self.convert_file(json_file, output_data, format)
        else:
            self.convert_file(input_data, output_data, format)

    def convert_file(self, input_data: str, output_data: str, format: str):
        items = self.iter_from_json_file(input_data)
        basename = os.path.splitext(os.path.basename(input_data))[0]
        output_data = os.path.join(output_data, basename)
        os.makedirs(output_data, exist_ok=True)
        convert_task(items, output_data, out_format=format)

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
            with open(json_file, "rb") as f:
                for task in ijson.items(f, "item", use_float=True):
                    for item in self.annotation_result_from_task(task):
                        if item is not None:
                            yield item

    def iter_from_dir(self, input_dir):
        for json_file in os.listdir(input_dir):
            if json_file[-5:] != ".json":
                continue
            json_file = os.path.join(input_dir, json_file)
            yield json_file

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
            yield []

        # skip cancelled annotations
        annotations = list(filter(lambda x: not (
            x.get("skipped", False) or x.get("was_cancelled", False)
        ), annotations))
        if not annotations:
            return None

        for annotation in annotations:
            yield annotation["result"]


def convert_parser(parser):
    parser.add_argument(
        "-i",
        "--input",
        dest="input",
        help="JSON file with annotations",
        required=True,
        action=ExpandFullPath,
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        help="Output file or directory (will be created if not exists)",
        default=os.path.join(os.path.dirname(__file__), "output"),
        action=ExpandFullPath,
    )
    parser.add_argument(
        "-f",
        "--format",
        dest="format",
        metavar="FORMAT",
        type=str,
        default="png",
        choices=["png", "numpy"],
        help="Converter format",
    )
