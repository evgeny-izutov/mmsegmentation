import argparse
import json
from os import makedirs, listdir
from os.path import join, exists, isfile
from shutil import copyfile
from typing import List, Tuple, Optional, Sequence, cast
from copy import copy

import cv2 as cv
import numpy as np


Contour = List[Tuple[float, float]]
OBJ_COLOR = "#e96115ff"
EMPTY_COLOR = "#85c4c1ff"


def get_project_config(project_name="seg"):
    pos_label = {"name": "object", "color": OBJ_COLOR, "group": "object", "is_empty": False}
    empty_label = {"name": "Empty", "color": EMPTY_COLOR, "group": "Empty", "is_empty": True}
    labels = [pos_label, empty_label]

    cfg = {
        "name": project_name,
        "pipeline": {
            "tasks": [
                {
                    "title": "Dataset",
                    "task_type": "dataset"
                }, 
                {
                    "title": "Segmentation task",
                    "task_type": "segmentation",
                    "labels": labels
                }
            ],
            "connections": [
                {
                    "to": "Segmentation task",
                    "from": "Dataset"
                }
            ]
        },
        "datasets": [
            {
                "name": project_name
            }
        ]
    }

    return cfg


def get_annot_from_polygons(polygons: List[List[Tuple]]):
    labels_info = [{"probability": 1.0, "name": "object", "color": OBJ_COLOR, "source": {"id": "N/A", "type": "N/A"}}]

    annotations = []
    for polygon in polygons:
        points = [{"x": point[0], "y": point[1]} for point in polygon]

        polygon_info = {
            "labels": copy(labels_info),
            "shape": {"type": "POLYGON", "points": points}
        }
        annotations.append(polygon_info)

    data = {
        "annotations": annotations,
        "kind": "annotation"
    }

    return data


def dump_to_json(data, output_file):
    assert isinstance(data, dict)

    with open(output_file, 'w') as output_stream:
        json.dump(data, output_stream)


def convert_annotation(in_dir, out_dir, extension):
    if not exists(out_dir):
        makedirs(out_dir)
    
    cut_symbols = len(extension) + 1

    for subdir in ['training', 'validation']:
        files = [f for f in listdir(join(in_dir, subdir)) if isfile(join(in_dir, subdir, f))]
        annot = [f for f in files if f.endswith(extension)]

        for annot_file in annot:
            mask = cv.imread(join(in_dir, subdir, annot_file), cv.IMREAD_GRAYSCALE)
            polygons = convert_mask_to_polygons(mask == 1)

            out_annot = get_annot_from_polygons(polygons)
            out_annot_file = join(out_dir, f'{subdir}_{annot_file[:-cut_symbols]}.json')
            dump_to_json(out_annot, out_annot_file)


def copy_images(in_dir, out_dir, extension):
    if not exists(out_dir):
        makedirs(out_dir)

    for subdir in ['training', 'validation']:
        files = [f for f in listdir(join(in_dir, subdir)) if isfile(join(in_dir, subdir, f))]
        images = [f for f in files if f.endswith(extension)]

        for image_file in images:
            copyfile(
                join(in_dir, subdir, image_file),
                join(out_dir, f'{subdir}_{image_file}')
            )


def get_subcontours(contour: Contour) -> List[Contour]:
    ContourInternal = List[Optional[Tuple[float, float]]]

    def find_loops(points: ContourInternal) -> List[Sequence[int]]:
        """
        For each consecutive pair of equivalent rows in the input matrix
        returns their indices.
        """
        _, inverse, count = np.unique(
            points, axis=0, return_inverse=True, return_counts=True
        )
        duplicates = np.where(count > 1)[0]
        indices = []
        for x in duplicates:
            y = np.nonzero(inverse == x)[0]
            for i, _ in enumerate(y[:-1]):
                indices.append(y[i : i + 2])
        return indices

    base_contour = cast(ContourInternal, copy(contour))

    # Make sure that contour is closed.
    if not np.array_equal(base_contour[0], base_contour[-1]):
        base_contour.append(base_contour[0])

    subcontours: List[Contour] = []
    loops = sorted(find_loops(base_contour), key=lambda x: x[0], reverse=True)
    for loop in loops:
        i, j = loop
        subcontour = base_contour[i:j]
        subcontour = list(x for x in subcontour if x is not None)
        subcontours.append(cast(Contour, subcontour))
        base_contour[i:j] = [None] * (j - i)

    subcontours = [i for i in subcontours if len(i) > 2]

    return subcontours


def convert_mask_to_polygons(mask):
    label_index_map = (mask.astype(int) * 255).astype(np.uint8)
    height, width = label_index_map.shape[:2]

    contours, hierarchies = cv.findContours(
        label_index_map, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE
    )

    polygons = []
    if hierarchies is not None:
        for contour, hierarchy in zip(contours, hierarchies[0]):
            if hierarchy[3] == -1:
                contour = list((point[0][0], point[0][1]) for point in contour)
                subcontours = get_subcontours(contour)

                for subcontour in subcontours:
                    points = [
                        (x / width, y / height) for x, y in subcontour
                    ]
                    polygons.append(points)
    return polygons


def parse_args():
    parser = argparse.ArgumentParser(description='Convert Kvasir dataset to SC REST API format')
    parser.add_argument('input_dir', help='the input folder')
    parser.add_argument('output_dir', help='the output folder')
    parser.add_argument('--project_name', type=str, default='kvasir-seg', help='The output project name')
    parser.add_argument('--image_ext', type=str, default='jpg', help='the extension of input images')
    parser.add_argument('--mask_ext', type=str, default='png', help='the extension of input masks')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    assert exists(args.input_dir)

    output_dir = join(args.output_dir, args.project_name)
    if not exists(output_dir):
        makedirs(output_dir)

    project_config = get_project_config(args.project_name)
    out_project_file = join(output_dir, 'project.json')
    dump_to_json(project_config, out_project_file)

    in_annot_dir = join(args.input_dir, 'annotations')
    out_annot_dir = join(output_dir, 'annotations')
    convert_annotation(in_annot_dir, out_annot_dir, args.mask_ext)

    in_images_dir = join(args.input_dir, 'images')
    out_images_dir = join(output_dir, 'images')
    copy_images(in_images_dir, out_images_dir, args.image_ext)


if __name__ == '__main__':
    main()
