def writeQML(labels, filename: str, colors: dict):
    out = ("<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n"
           "<qgis version=\"3.28.0\" styleCategories=\"Symbology\">\n"
           " <renderer-v2 type=\"categorizedSymbol\" attr=\"class_id\" forceraster=\"0\" "
           "symbollevels=\"0\" enableorderby=\"0\">\n  <categories>\n")
    for label in labels:
        if label not in colors:
            continue
        out += f"   <category label=\"{label}\" color=\"{colors[label]}\"/>\n"
    out += "  </categories>\n </renderer-v2>\n</qgis>\n"
    with open(filename, "w") as f:
        f.write(out)