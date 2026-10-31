import numpy as np
import rasterio.features
from shapely.geometry import shape
from shapely import make_valid


def table_geojson_from_annotation(layer: np.ndarray, label: str, transformer):
    table = []
    mask = (layer > 0).astype(np.uint8)
    shapes = rasterio.features.shapes(
        mask, mask=mask, transform=transformer._transformer
    )
    for geom, _ in shapes:
        p = shape(geom)
        if not p.is_valid:
            p = make_valid(p, method="structure", keep_collapsed=False)
        if p.is_empty:
            continue
        table.append({"class": label, "geometry": p})
    return table
