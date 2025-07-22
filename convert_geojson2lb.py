import os
import cv2
import json
import argparse
import rasterio
import numpy as np
import pandas as pd
from glob import glob
import geopandas as gpd
from shapely.geometry import Polygon, GeometryCollection, MultiPolygon
from rasterio.transform import AffineTransformer
from imports.brush import mask2annotation
from imports.utils import new_task


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', "--path", type=str, required=True)
    parser.add_argument('-l', "--label", type=str, required=True)
    parser.add_argument('--default_image_root_url', type=str, default='4_3')
    parser.add_argument('--out_type', type=str, default='annotations')
    parser.add_argument("--save_geojson", action='store_true')
    parser.add_argument("--save_png", action='store_true')
    return parser.parse_args()


def get_label_id(label: str):
    for i, x in label2id.items():
        if i in label:
            return x


def get_coords(poly: Polygon, transformer: AffineTransformer):
    new_coords = []
    for x, y in poly.exterior.coords[:-1]:
        y, x = transformer.rowcol(x, y)
        new_coords.append([x, y])
    return np.array(new_coords, np.int32)


def delete_labels(labels):
    # Удалить из поля леса
    labels[2] = cv2.bitwise_and(cv2.bitwise_not(labels[0]), labels[2])
    # Удалить из поля кустарники
    labels[2] = cv2.bitwise_and(cv2.bitwise_not(labels[1]), labels[2])
    # Удалить из поля строения
    labels[2] = cv2.bitwise_and(cv2.bitwise_not(labels[4]), labels[2])
    # Удалить из леса вырубку
    labels[0] = cv2.bitwise_and(cv2.bitwise_not(labels[5]), labels[0])
    # Удалить из леса заболоченные
    labels[0] = cv2.bitwise_and(cv2.bitwise_not(labels[6]), labels[0])
    # Удалить из заболоченного воду
    labels[6] = cv2.bitwise_and(cv2.bitwise_not(labels[3]), labels[6])
    # Удалить Поле из с/х
    labels[7] = cv2.bitwise_and(cv2.bitwise_not(labels[2]), labels[7])


if __name__ == '__main__':
    args = get_args()
    label2id = {
        'tree_row': 0,
        'tree_group': 0,
        'wood': 0,
        'Лес': 0,
        'scrub': 1,
        'кустарник': 1,
        'heath': 2,
        'grassland': 2,
        'Луг, поле': 2,
        'natural_gully': 2,
        'Поле': 2,
        'water': 3,
        'Вода': 3,
        'building': 4,
        'строения': 4,
        'landuse_forest': 5,
        'вырубка': 5,
        'wetland': 6,
        'заболоченный': 6,
        'landuse_farmland': 7,
        'сельхоз': 7,
        'сельскохозяйственные': 7,
        'территория фермы': 7,
        'Заболоченный луг': 8
    }
    id2label = {
        0: 'Лес',
        1: 'кустарник',
        2: 'Поле',
        3: 'Вода',
        4: 'строения',
        5: 'вырубка',
        6: 'заболоченный',
        7: "сельскохозяйственные",
        8: "Заболоченный луг"
    }
    geojson = glob(args.label + '/*.geojson')

    for tiff_path in glob(args.path + '/*.tif'):

        json_path = os.path.splitext(tiff_path)[0] + '.json'
        if os.path.exists(json_path):
            continue

        with rasterio.open(tiff_path) as src:
            img = src.read(4)
            crs = src.crs
            transformer = AffineTransformer(src.transform)

        h, w = img.shape
        _, thresh = cv2.threshold(img, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=lambda x: cv2.contourArea(x))
        new_p = [transformer.xy(x[1], x[0]) for x in cnt.reshape(-1, 2)]
        tiff_gdf = gpd.GeoDataFrame({'geometry': [Polygon(new_p)]}, crs=crs)
        folder = os.path.split(os.path.splitext(tiff_path)[0])[1]
        labels = np.zeros((9, h, w), dtype=np.uint8)

        for file in geojson:
            gdf = gpd.read_file(file)
            gdf_crs = gdf.crs
            if gdf.crs != crs:
                gdf = gdf.to_crs(crs)
            clipped_gdf = gpd.clip(gdf, tiff_gdf)
            clipped_gdf = clipped_gdf.reset_index(drop=True)
            for i in clipped_gdf.index:
                geom = clipped_gdf.loc[i, 'geometry']
                if isinstance(geom, GeometryCollection) or isinstance(geom, MultiPolygon):
                    add_polygon = False
                    for g in geom.geoms:
                        if isinstance(g, Polygon):
                            if not add_polygon:
                                clipped_gdf.loc[i, 'geometry'] = g
                                add_polygon = True
                            else:
                                clipped_gdf = pd.concat(
                                    [clipped_gdf, gpd.GeoDataFrame([clipped_gdf.iloc[i]], crs=crs)],
                                    ignore_index=True
                                )
                                clipped_gdf.loc[len(clipped_gdf), 'geometry'] = g

            if len(clipped_gdf) == 0:
                continue

            label_id = get_label_id(file)
            for geom in clipped_gdf.geometry:
                if isinstance(geom, Polygon):
                    cv2.fillPoly(labels[label_id], [get_coords(geom, transformer)], (255))

            if args.save_geojson:
                if gdf_crs != crs:
                    clipped_gdf = clipped_gdf.to_crs(gdf_crs)
                head, tail = os.path.split(file)
                to_folder = head + '/' + folder
                os.makedirs(to_folder, exist_ok=True)
                file = to_folder + '/' + tail
                print(file)
                clipped_gdf.to_file(file, driver='GeoJSON')

        delete_labels(labels)

        jpg_file = os.path.splitext(tiff_path)[0] + '.jpg'
        if not os.path.exists(jpg_file):
            cv2.imwrite(jpg_file, cv2.imread(tiff_path, cv2.IMREAD_GRAYSCALE))
        image_root_url = '/data/local-files/?d=' + args.default_image_root_url
        image_root_url += "" if image_root_url.endswith("/") else "/"

        head, _ = os.path.splitext(jpg_file)
        jpg_file = os.path.basename(jpg_file)
        task = new_task(args.out_type, image_root_url, jpg_file)
        for i, mask in enumerate(labels):
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
