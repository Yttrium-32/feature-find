from modules.colordescriptor import ColorDescriptor
from modules.index import Indexer
from search_gui.models import ImageFeatures

import numpy
import csv

import glob
import os


class Searcher:
    def __init__(self, photo_dir: str, clr_dsc: ColorDescriptor) -> None:
        self.photo_dir = photo_dir
        # self.build_index(clr_dsc)

    def build_index(self, clr_dsc: ColorDescriptor) -> None:
        indexed_names = set(ImageFeatures.objects.values_list('image_name', flat=True))
        all_images = set(os.path.basename(p) for p in glob.glob(f"{self.photo_dir}/*.jpg"))
        missing_images = all_images - indexed_names

        if missing_images:
            print(f"INFO: Building index for {len(missing_images)} new images...")
            indexer = Indexer(clr_dsc)
            indexer.index_images(self.photo_dir)

    def chi2_distance(self, hist_a, hist_b, eps = 1e-10):
        d = 0.5 * numpy.sum([
                ((a - b) ** 2) / (a + b + eps)
                for (a,b) in zip(hist_a, hist_b)
        ])

        return d

    def search(self, query_features, limit: int = 10):
        results = dict()

        for image_feature in ImageFeatures.objects.all():
            features = image_feature.features
            d = self.chi2_distance(features, query_features)
            results[image_feature.image_name] = d

        results = sorted(results.items(), key=lambda item: item[1])

        return results[:limit]

