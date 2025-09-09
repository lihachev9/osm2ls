import argparse
from converter import Converter, convert_parser
from imports import yolo as import_yolo, coco as import_coco, segm as import_segm


def exports(args):
    converter = Converter()
    converter.convert(args.input, args.output, args.format)


def imports(args):
    if args.import_format == "yolo":
        import_yolo.convert_yolo_to_ls(
            input_dir=args.input,
            out_file=args.output,
            to_name=args.to_name,
            from_name=args.from_name,
            out_type=args.out_type,
            image_root_url=args.image_root_url,
            image_ext=args.image_ext,
            task_type=args.task_type
        )
    elif args.import_format == "coco":
        import_coco.convert_coco_to_ls(
            input_file=args.input,
            out_file=args.output,
            to_name=args.to_name,
            from_name=args.from_name,
            out_type=args.out_type,
            image_root_url=args.image_root_url,
        )
    elif args.import_format == "segm":
        import_segm.convert_segm_to_ls(
            input_file=args.input,
            out_file=args.output,
            to_name=args.to_name,
            from_name=args.from_name,
            out_type=args.out_type,
            image_root_url=args.image_root_url
        )
    else:
        raise NotImplementedError()


def get_all_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = False

    # Export
    parser_export = subparsers.add_parser(
        "export",
        help="Converter from Label Studio JSON annotations to external formats",
    )
    convert_parser(parser_export)

    # Import
    parser_import = subparsers.add_parser(
        "import",
        help="Converter from external formats to Label Studio JSON annotations",
    )
    import_format = parser_import.add_subparsers(dest="import_format")
    import_yolo.add_parser(import_format)
    import_coco.add_parser(import_format)
    import_segm.add_parser(import_format)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_all_args()
    if args.command == "export":
        exports(args)
    elif args.command == "import":
        imports(args)
    else:
        print("Please, specify command: 'export' or 'import'")
