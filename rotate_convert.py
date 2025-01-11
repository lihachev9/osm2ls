import argparse
import os
import cv2
import numpy as np
from PIL import Image
import rasterio as rs
import geopandas as gpd
from rasterio.transform import AffineTransformer
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Polygon, box, Point, LineString


def find_p0(img: np.ndarray):
    h, w = img.shape
    for i in range(w):
        for j in range(h):
            if img[j][i] != 0:
                return i, j


def find_p1(img: np.ndarray):
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            if img[i][j] != 0:
                return j, i


def find_p2(img: np.ndarray):
    h, w = img.shape
    for i in range(h - 1, -1, -1):
        for j in range(w):
            if img[i][j] != 0:
                return j, i


def find_all_p(img: np.ndarray):
    return find_p0(img), find_p1(img), find_p2(img)


class Affine:
    def __init__(self, a, offset):
        self.a = a
        self.offset = offset
        self.a_inv = np.linalg.inv(a)

    def invert_affine_transform(self, src: np.ndarray, h: float, w: float) -> np.ndarray:
        M, N = src.shape
        points = np.mgrid[0:np.uint16(N), 0:np.uint16(M)].reshape((2, M*N))
        new_points = self.a.dot(points).round().astype('int32')
        new_points[0] += self.offset[0]
        new_points[1] += self.offset[1]
        x, y = new_points.reshape((2, M, N), order='F')
        indices = x + N * y
        return np.take(src, indices, mode='wrap')[:int(h) + 1, :int(w) + 1]

    def affine_transform(self, x: int, y: int):
        new_points  = self.a_inv.dot([x, y])
        new_points -= self.a_inv.dot(self.offset)
        return new_points.round().astype('int32')


def accumulate_cut(img: np.ndarray, min_percent=2, max_percent=98) -> np.ndarray:
    lo, hi = np.percentile(img, (min_percent, max_percent))
    res_img = (img.astype(float) - lo) / (hi-lo)

    #Multiply by 255, clamp range to [0, 255] and convert to uint8
    res_img = np.maximum(np.minimum(res_img*255, 255), 0).astype(np.uint8)
    return res_img


def find_box(coords) -> np.ndarray:
    coords = np.array(coords, 'int32').reshape(-1, 1, 2)
    rect = cv2.minAreaRect(coords)
    box = cv2.boxPoints(rect).astype('int32')
    return box


def get_annotations(geometry, transformer, affine_transform):
    annotations = []
    for item in geometry:
        coords = []
        for poly in item.geoms:
            for x, y in poly.exterior.coords[:-1]:
                y, x = transformer.rowcol(x, y)
                x, y = affine_transform(x, y)
                coords.append((x, y))
        if coords == []:
            continue
        coords = find_box(coords)
        annotations.append(Polygon(coords))
    return annotations


def start_points(size: int, split_size: int, overlap=0.0):
    points = [0]
    stride = int(split_size * (1-overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            if split_size == size:
                break
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points


def get_coords(obbox: Polygon) -> np.ndarray:
    coords = np.array(obbox.exterior.coords[:-1])
    tolerance = 1
    while len(coords) > 4:
        coords = np.array(obbox.simplify(tolerance).exterior.coords[:-1])
        tolerance += 1
    return coords


def split(img,
          annotations,
          part_name: str,
          parts_h=2, parts_w=2,
          output_dir='',
          images_dir='images',
          label_dir='labels'):
    images_dir = os.path.join(output_dir, images_dir)
    label_dir = os.path.join(output_dir, label_dir)
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    part_name = os.path.basename(part_name)
    img_pil = Image.fromarray(img)
    w, h = img_pil.size
    target_w = int(w / parts_w + w % parts_w)
    target_h = int(h / parts_h + h % parts_h)
    X_points = start_points(w, target_w)
    Y_points = start_points(h, target_h)

    for i, left in enumerate(X_points):
        for j, top in enumerate(Y_points):
            right = left + target_w
            bottom = top + target_h
            part_box = box(left, top, right, bottom)
            new_annotations = []
            for obbox in annotations:
                if part_box.intersects(obbox):
                    coords = np.array(obbox.exterior.coords[:-1])
                    # Получение координат нового obbox
                    coords[:, 0] = (coords[:, 0] - left) / target_w
                    coords[:, 1] = (coords[:, 1] - top) / target_h

                    # Проверка на выход за границы
                    if (0 > coords).any() or (coords > 1).any():
                        # Вычисление нового obbox
                        new_obbox: BaseGeometry = obbox.intersection(part_box)
                        if isinstance(new_obbox, Point) or isinstance(new_obbox, LineString):
                            continue
                        coords = get_coords(new_obbox)
                        coords[:, 0] = (coords[:, 0] - left) / target_w
                        coords[:, 1] = (coords[:, 1] - top) / target_h
                    if len(coords) < 4:
                        continue

                    formatted_coords = [f"{coord:.6g}" for coord in coords.reshape(-1)]
                    new_annotations.append(f"{0} {' '.join(formatted_coords)}\n")

            # Сохранение новых аннотаций
            annotation_name = f"{os.path.splitext(part_name)[0]}_{left}_{top}.txt"
            with open(os.path.join(label_dir, annotation_name), 'w') as f:
                f.writelines(new_annotations)

            image_name = f"{os.path.splitext(part_name)[0]}_{left}_{top}.jpg"
            img_pil.crop((left, top, right, bottom)).save(os.path.join(images_dir, image_name))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', "--path", type=str, required=True)
    parser.add_argument('-l', "--label", type=str, required=True)
    parser.add_argument('-o', "--output", type=str, default="")
    parser.add_argument("--parts_w", type=int, default=2)
    parser.add_argument("--parts_h", type=int, default=2)
    return parser.parse_args()


if __name__=='__main__':
    args = get_args()
    gdf = gpd.read_file(args.label)
    with rs.open(args.path) as src:
        img = src.read(1)
        transformer = AffineTransformer(src.transform)
    p = find_all_p(img)
    w = np.sqrt((p[1][0]-p[0][0])**2+(p[1][1]-p[0][1])**2)
    h = np.sqrt((p[2][0]-p[0][0])**2+(p[2][1]-p[0][1])**2)
    pad_h = abs(img.shape[0] - int(h))
    img = np.pad(img, ((0, pad_h), (0, 0)), mode='constant', constant_values=0)
    a = np.array([[(p[1][0]-p[0][0])/w, (p[2][0]-p[0][0])/h],
                  [(p[1][1]-p[0][1])/w, (p[2][1]-p[0][1])/h]])
    offset = p[0]
    affine = Affine(a, offset)
    print('p =', p)
    print('a =', a)
    img = affine.invert_affine_transform(img, h, w)
    img = accumulate_cut(img)
    annotations = get_annotations(gdf.geometry, transformer, affine.affine_transform)
    split(img, annotations, args.path, args.parts_w, args.parts_h, args.output)
