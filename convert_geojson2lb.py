import os
import cv2
import csv
import json
import argparse
import rasterio
import numpy as np
from glob import glob
from PIL import Image
import geopandas as gpd
from shapely.geometry import Polygon, LineString
from shapely.errors import GEOSException
from rasterio.transform import AffineTransformer
from brush import mask2annotation
from imports.utils import new_task
from rotate_convert import start_points


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', "--path", type=str, required=True)
    parser.add_argument('-l', "--label", type=str, required=True)
    parser.add_argument('--default_image_root_url', type=str, default='4_3')
    parser.add_argument('--out_type', type=str, default='annotations')
    parser.add_argument("--save_geojson", action='store_true')
    parser.add_argument("--save_png", action='store_true')
    parser.add_argument("--parts_w", type=int, default=1)
    parser.add_argument("--parts_h", type=int, default=1)
    return parser.parse_args()


def get_label_id(label2id, label: str):
    for i, x in label2id.items():
        if i in label:
            return x


def get_coords(coords, transformer: AffineTransformer):
    new_coords = []
    for x, y in coords:
        y, x = transformer.rowcol(x, y)
        new_coords.append([x, y])
    return np.array(new_coords, np.int32)


def delete_labels(labels, id2label: dict, delete_label: dict):
    label2id = {v: k for k, v in id2label.items()}
    for k, v_set in delete_label.items():
        if k not in label2id:
            print(f"{k} нет на изображение, поэтому пропускаем удаление для него")
            continue
        bitwise = cv2.bitwise_not(labels[label2id[k]])
        for v in v_set:
            if v not in label2id:
                print(f"{v} нет на изображение, поэтому пропускаем удаление из него")
                continue
            labels[label2id[v]] = cv2.bitwise_and(bitwise, labels[label2id[v]])


def get_id2label(data):
    id2label1 = dict()
    count = 0
    idkeys = set()
    for x in data:
        t = x[1]
        if t not in idkeys:
            idkeys.add(t)
            id2label1[count] = t
            count += 1
    return id2label1


def get_label2id(data, id2label):
    reverse_id2label = {v: k for k, v in id2label.items()}
    label2id = dict()
    for x in data:
        label2id[x[0]] = reverse_id2label[x[1]]
    return label2id


def get_delete_label() -> dict:
    delete_label = {}
    with open("label2studio.csv") as f:
        reader = csv.reader(f)
        for rows in reader:
            if len(rows) != 2:
                continue
            k, v = rows
            if k not in delete_label:
                delete_label[k] = set()
            delete_label[k].add(v)
    return delete_label


if __name__ == '__main__':
    args = get_args()
    parts_w = args.parts_w
    parts_h = args.parts_h
    reize = parts_w != 1 or parts_h != 1
    with open("label2studio.csv") as f:
        reader = csv.reader(f)
        label2studio = {rows[0]: rows[1] for rows in reader}
    delete_label = get_delete_label()
    id2label = get_id2label(label2studio)
    label2id = get_label2id(label2studio, id2label)
    geojson = glob(args.label + '/*.geojson')

    for tiff_path in glob(args.path + '/*.tif'):

        json_path = os.path.splitext(tiff_path)[0] + '.json'
        if os.path.exists(json_path):
            continue

        with rasterio.open(tiff_path) as src:
            img = src.read(1)
            crs = src.crs
            transformer = AffineTransformer(src.transform)

        h, w = img.shape
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=lambda x: cv2.contourArea(x))
        # cnt = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        new_p = [transformer.xy(x[1], x[0]) for x in cnt.reshape(-1, 2)]
        tiff_gdf = gpd.GeoDataFrame({'geometry': [Polygon(new_p)]}, crs=crs)
        folder = os.path.split(os.path.splitext(tiff_path)[0])[1]
        labels = np.zeros((len(id2label), h, w), dtype=np.uint8)

        for file in geojson:
            gdf = gpd.read_file(file)
            gdf_crs = gdf.crs
            if gdf.crs != crs:
                gdf = gdf.to_crs(crs)

            try:
                clipped_gdf = gpd.clip(gdf, tiff_gdf).explode(ignore_index=True)
            except GEOSException:
                print(f'Cannot clip {file} with {tiff_path}')
                continue

            if len(clipped_gdf) == 0:
                continue

            label_id = get_label_id(label2id, file)
            if label_id is None:
                print(f'Unknown label: {file}')
                continue

            for geom in clipped_gdf.geometry:
                if isinstance(geom, Polygon):
                    coords = geom.exterior.coords[:-1]
                    cv2.fillPoly(labels[label_id], [get_coords(coords, transformer)], (255))
                elif isinstance(geom, LineString):
                    coords = geom.coords
                    cv2.polylines(labels[label_id], [get_coords(coords, transformer)], False, (255), 10)

            if args.save_geojson:
                if gdf_crs != crs:
                    clipped_gdf = clipped_gdf.to_crs(gdf_crs)
                head, tail = os.path.split(file)
                to_folder = head + '/' + folder
                os.makedirs(to_folder, exist_ok=True)
                file = to_folder + '/' + tail
                print(file)
                clipped_gdf.to_file(file, driver='GeoJSON')

        delete_labels(labels, id2label, delete_label)

        target_w = int(w / parts_w + w % parts_w)
        target_h = int(h / parts_h + h % parts_h)
        X_points = start_points(w, target_w)
        Y_points = start_points(h, target_h)
        jpg_file = os.path.splitext(tiff_path)[0] + '.jpg'

        for left in X_points:
            for top in Y_points:
                if reize:
                    jpg_file = os.path.splitext(tiff_path)[0] + f'_{left}_{top}.jpg'
                    json_path = os.path.splitext(jpg_file)[0] + '.json'
                if not os.path.exists(jpg_file):
                    Image.fromarray(img[top:top + target_h, left:left + target_w]).save(jpg_file)
                image_root_url = '/data/local-files/?d=' + args.default_image_root_url
                image_root_url += "" if image_root_url.endswith("/") else "/"

                head, _ = os.path.splitext(jpg_file)
                task = new_task(args.out_type, image_root_url, os.path.basename(jpg_file))
                for i, mask in enumerate(labels):
                    mask = mask[top:top + target_h, left:left + target_w]
                    if set(np.unique(mask)) == {0}:
                        continue
                    name = id2label[i]
                    if args.save_png:
                        os.makedirs(head + '_label_png', exist_ok=True)
                        cv2.imwrite(head + '_label_png/' + name + '.png', mask)
                    annotation = mask2annotation(
                        mask=mask,
                        label_name=name,
                        from_name='tag',
                        to_name='image'
                    )
                    task[args.out_type][0]['result'].append(annotation)

                tasks = [task]
                with open(json_path, "w") as f:
                    json.dump(tasks, f)
