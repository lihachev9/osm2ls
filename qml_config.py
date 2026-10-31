import csv


def writeQML(labels, filename):
    colors = {}
    with open("color_labels.csv") as f:
        reader = csv.reader(f)
        for rows in reader:
            if len(rows) != 2:
                continue
            k, v = rows
            if k in labels:
                colors[k] = v
    out = ("<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n"
           "<qgis version=\"3.28.0\" styleCategories=\"Symbology\">\n"
           " <renderer-v2 type=\"categorizedSymbol\" attr=\"class_id\" forceraster=\"0\" "
           "symbollevels=\"0\" enableorderby=\"0\">\n  <categories>\n")
    for k, v in colors.items():
        out += f"   <category label=\"{k}\" color=\"{v}\"/>\n"
    out += "  </categories>\n </renderer-v2>\n</qgis>\n"
    with open(filename, "w") as f:
        f.write(out)