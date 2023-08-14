import os.path as osp
import json
import numpy as np
import PIL.Image
from pathlib import Path


def lblsave(filename, lbl):
    import imgviz

    if osp.splitext(filename)[1] != ".png":
        filename += ".png"
    # Assume label ranses [-1, 254] for int32,
    # and [0, 255] for uint8 as VOC.
    if lbl.min() >= -1 and lbl.max() < 255:
        lbl_pil = PIL.Image.fromarray(lbl.astype(np.uint8), mode="P")
        colormap = imgviz.label_colormap()
        lbl_pil.putpalette(colormap.flatten())
        lbl_pil.save(filename)
    else:
        raise ValueError(
            "[%s] Cannot save the pixel-wise class label as PNG. "
            "Please consider using the .npy format." % filename
        )


def label_to_path(json_file, ext=".jpg"):
    with open(json_file, "r", encoding="utf-8") as f:
        ann = json.load(f)
    info = list()
    for shape in ann["shapes"]:
        info.append(
            {
                "label": shape["label"],
                "filename": Path(json_file.replace(".json", ext)).as_posix(),
            }
        )
    return info
