"""The PEAR-eLite logo from its pixel cells: SVGs for light and dark pages (one rect per horizontal run of one color,
shape-rendering crispEdges, so it stays sharp at any size) and a PNG in which every cell is a whole number of pixels.

usage: python tools/assets/build_logo.py docs/assets/pear_elite_logo_cells.json docs/assets
"""
import json, sys
from PIL import Image
cells_json, out_dir = sys.argv[1:3]
d = json.load(open(cells_json))
PEAR_CELL, TEXT_CELL = 9, 4          # 27.5 px : 12.25 px in the source, about 2.25 : 1
PX_PER_UNIT = 27.5 / PEAR_CELL       # source pixels per SVG unit
def norm(cells):
    ks = {tuple(map(int, k.split(","))): tuple(v) for k, v in cells.items()}
    x0 = min(k[0] for k in ks); y0 = min(k[1] for k in ks)
    return {(i - x0, j - y0): c for (i, j), c in ks.items()}
pear, text = norm(d["pear"]), norm(d["text"])
# placement in SVG units, from the source's bounding boxes (pear left 190, text left 77 / top 485)
PEAR_X, PEAR_Y = round(190 / PX_PER_UNIT), 0
TEXT_X, TEXT_Y = round(77 / PX_PER_UNIT), round(485 / PX_PER_UNIT)
def runs(cells):
    out = []
    for j in sorted({k[1] for k in cells}):
        row = sorted(k[0] for k in cells if k[1] == j)
        i = 0
        while i < len(row):
            s = row[i]; c = cells[(s, j)]; e = s
            while i + 1 < len(row) and row[i + 1] == e + 1 and cells[(row[i + 1], j)] == c:
                i += 1; e = row[i]
            out.append((s, j, e - s + 1, c)); i += 1
    return out
pear_w = max(k[0] for k in pear) + 1; text_w = max(k[0] for k in text) + 1; text_h = max(k[1] for k in text) + 1
M = 6
x_min = min(PEAR_X, TEXT_X) - M; x_max = max(PEAR_X + pear_w * PEAR_CELL, TEXT_X + text_w * TEXT_CELL) + M
y_min = PEAR_Y - M; y_max = TEXT_Y + text_h * TEXT_CELL + M
W, H = x_max - x_min, y_max - y_min
hexc = lambda c: "#%02x%02x%02x" % c
def svg(text_color):
    rects = []
    for i, j, n, c in runs(pear):
        rects.append(f'<rect x="{PEAR_X + i * PEAR_CELL - x_min}" y="{PEAR_Y + j * PEAR_CELL - y_min}" width="{n * PEAR_CELL}" height="{PEAR_CELL}" fill="{hexc(c)}"/>')
    rects.append(f'<g fill="{text_color}">')
    for i, j, n, c in runs(text):
        rects.append(f'<rect x="{TEXT_X + i * TEXT_CELL - x_min}" y="{TEXT_Y + j * TEXT_CELL - y_min}" width="{n * TEXT_CELL}" height="{TEXT_CELL}"/>')
    rects.append("</g>")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W * 2}" height="{H * 2}" '
            f'shape-rendering="crispEdges" role="img" aria-label="PEAR-eLite">\n<title>PEAR-eLite</title>\n' + "\n".join(rects) + "\n</svg>\n")
open(f"{out_dir}/pear_elite_logo_light.svg", "w").write(svg("#000000"))
open(f"{out_dir}/pear_elite_logo_dark.svg", "w").write(svg("#f0f6fc"))
# raster: every SVG unit -> S pixels, so each cell is a whole number of pixels
S = 4
im = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
for cells, ox, oy, cs, fixed in ((pear, PEAR_X, PEAR_Y, PEAR_CELL, None), (text, TEXT_X, TEXT_Y, TEXT_CELL, (0, 0, 0))):
    for (i, j), c in cells.items():
        x = (ox + i * cs - x_min) * S; y = (oy + j * cs - y_min) * S
        im.paste(tuple(fixed or c) + (255,), (x, y, x + cs * S, y + cs * S))
im.save(f"{out_dir}/pear_elite_logo.png", optimize=True)
print("viewBox", W, H, "png", im.size, "rects", len(runs(pear)) + len(runs(text)))
