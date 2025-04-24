
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
    parser.add_argument('-l', "--label", type=str)
    parser.add_argument("--save_geojson", action='store_true')
    parser.add_argument("--save_png", action='store_true')
    return parser.parse_args()


def get_label_id(label):
    for i, x in label2id.items():
        if i in label:
            return x


if __name__ == '__main__':
    args = get_args()
    label2id = {
        'tree_row': 0,
        'wood': 0,
        'scrub': 1,
        'heath': 2,
        'grassland': 2,
        'natural_gully': 2,
        'landuse_farmland': 2,
        'water': 3,
        'building': 4,
        'landuse_forest': 5,
        'wetland': 6
    }
    id2label = {
        0: 'лес',
        1: 'кустарники',
        2: 'поле',
        3: 'вода',
        4: 'строения',
        5: 'вырубка',
        6: 'заболоченный'
    }
    default_image_root_url = "/data/local-files/?d=3b"
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

        new_p = [transformer.xy(x[1], x[0]) for x in contours[0].reshape(-1, 2)]
        tiff_gdf = gpd.GeoDataFrame({'geometry': [Polygon(new_p)]}, crs=crs)
        folder = os.path.split(os.path.splitext(tiff_path)[0])[1]
        labels = np.zeros((7, h, w), dtype=np.uint8)

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
                                clipped_gdf = pd.concat([clipped_gdf, gpd.GeoDataFrame([clipped_gdf.iloc[i]], crs=crs)], ignore_index=True)
                                clipped_gdf.loc[len(clipped_gdf), 'geometry'] = g

            if len(clipped_gdf) == 0:
                continue

            label_id = get_label_id(file)
            for geom in clipped_gdf.geometry:
                if isinstance(geom, Polygon):
                    coords = []
                    for x in geom.exterior.coords[:-1]:
                        x, y = transformer.rowcol(x[0], x[1])
                        coords.append([y, x])
                    cv2.fillPoly(labels[label_id], [np.array(coords, 'int64')], (255))

            if args.save_geojson:
                if gdf_crs != crs:
                    clipped_gdf = clipped_gdf.to_crs(gdf_crs)
                head, tail =  os.path.split(file)
                to_folder = head + '/' + folder
                os.makedirs(to_folder, exist_ok=True)
                file = to_folder + '/' + tail
                clipped_gdf.to_file(file, driver='GeoJSON')

        labels[2] = cv2.bitwise_and(cv2.bitwise_not(labels[0]), labels[2])
        labels[2] = cv2.bitwise_and(cv2.bitwise_not(labels[1]), labels[2])
        labels[2] = cv2.bitwise_and(cv2.bitwise_not(labels[4]), labels[2])
        labels[0] = cv2.bitwise_and(cv2.bitwise_not(labels[5]), labels[0])
        labels[0] = cv2.bitwise_and(cv2.bitwise_not(labels[6]), labels[0])
        labels[6] = cv2.bitwise_and(cv2.bitwise_not(labels[3]), labels[6])

        jpg_file = os.path.splitext(tiff_path)[0] + '.jpg'
        if not os.path.exists(jpg_file):
            cv2.imwrite(jpg_file, cv2.imread(tiff_path, cv2.IMREAD_GRAYSCALE))
        image_root_url=default_image_root_url
        image_root_url += "" if image_root_url.endswith("/") else "/"

        out_type = "annotations"
        jpg_file = os.path.split(jpg_file)[-1]
        task = new_task(out_type, image_root_url, jpg_file)
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
            task[out_type][0]['result'].append(annotation)

        tasks = [task]
        with open(json_path, "w") as f:
            json.dump(tasks, f)
