from image_matcher.settings import BASE_DIR, MEDIA_ROOT, MEDIA_URL
from modules.colordescriptor import ColorDescriptor
from modules.index import Indexer

from sklearn.metrics.pairwise import cosine_similarity

import cv2
import numpy

import argparse
import csv
import os

class Searcher:
    def __init__(self, photo_dir: str, index_path: str, bins: tuple[int, int, int]) -> None:
        self.index_path = index_path
        self.photo_dir = photo_dir
        self.init_index(bins)

    def init_index(self, bins: tuple[int, int, int]) -> None:
        indexer = Indexer(bins)

        if not os.path.isfile(self.index_path):
            indexer.index_images(self.photo_dir, self.index_path)

    def chi2_distance(self, hist_a, hist_b, eps = 1e-10):
        d = 0.5 * numpy.sum([
                ((a - b) ** 2) / (a + b + eps)
                for (a,b) in zip(hist_a, hist_b)
        ])

        return d

    def search(self, query_features, limit: int = 10):
        results = dict()

        with open(self.index_path) as index_file:
            csv_file = csv.reader(index_file)

            for row in csv_file:
                features = [float(x) for x in row[1:]]

                vec_db = numpy.array(features).reshape(1, -1)
                vec_query = numpy.array(query_features).reshape(1, -1)

                d = cosine_similarity(vec_db, vec_query)[0][0]

                results[row[0]] = d

        results = sorted(results.items(), key=lambda item: item[1], reverse=True)

        return results[:limit]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
            "-q", "--query",
            required = True,
            help = "Path to the image to be queried"
    )

    args = vars(ap.parse_args())

    query = cv2.imread(args["query"])
    matched_photos = []

    bins = (8, 12, 3)
    clr_dsc = ColorDescriptor(bins)
    searcher = Searcher(
            MEDIA_ROOT.__str__(),
            BASE_DIR / "index.csv",
            bins
    )

    features = clr_dsc.describe(query)
    results = searcher.search(features)

    print(f"INFO: index path: {BASE_DIR / 'index.csv'}")
    for (result_id, score) in results:
        print(f"INFO: {result_id=}, {score=}")
        matched_photos.append(f"{MEDIA_URL}{result_id}")

    print(f"DEBUG: {matched_photos=}")

