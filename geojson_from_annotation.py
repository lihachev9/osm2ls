import numpy as np
import rasterio.features
from shapely.geometry import shape
from shapely import make_valid


def table_from_annotation(layer: np.ndarray, label: str, transform, colors: dict):
    table = []
    mask = (layer > 0).astype(np.uint8)
    shapes = rasterio.features.shapes(
        mask, mask=mask, transform=transform
    )
    for geom, _ in shapes:
        p = shape(geom)
        if not p.is_valid:
            p = make_valid(p, method="structure", keep_collapsed=False)
        if p.is_empty:
            continue
        row = {"class": label, "geometry": p}
        if label in colors:
            row["fill"] = colors[label]
        table.append(row)
    return table
