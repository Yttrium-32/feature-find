from image_matcher.settings import BASE_DIR, INDEX_FILE, MEDIA_ROOT, MEDIA_URL
from modules.feature_extractor import FeatureExtractor
from modules.index import Indexer

from sklearn.metrics.pairwise import cosine_similarity
import cv2
import numpy
import h5py

from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import os

class Searcher:
    def __init__(self, photo_dir: str, index_path: str) -> None:
        self.index_path = index_path
        self.photo_dir = photo_dir
        self.init_index()

    def init_index(self) -> None:
        indexer = Indexer()

        if not os.path.isfile(self.index_path):
            indexer.index_images(self.photo_dir, self.index_path)

    def compute_cos_sim(self, image_name, image_features, query_features):
        vec_image = numpy.array(image_features).reshape(1, -1)
        vec_query = numpy.array(query_features).reshape(1, -1)

        sim = cosine_similarity(vec_image, vec_query)[0][0]

        if sim > 0.5:
            return (image_name, sim)
        return None


    def search(self, query_features, limit: int = 10):
        results = []

        with h5py.File(self.index_path, "r") as index_file:
            all_features = index_file["features"][:]
            image_names = index_file["image_names"][:]

            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        self.compute_cos_sim,
                        image_names[idx],
                        all_features[idx],
                        query_features
                    )
                    for idx in range(all_features.shape[0])
                ]
                for future in as_completed(futures):
                    res = future.result()
                    if res:
                        results.append(res)

        results.sort(key=lambda item: item[1], reverse=True)
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
    feature_extractor = FeatureExtractor()
    searcher = Searcher(
            MEDIA_ROOT.__str__(),
            INDEX_FILE,
    )

    features = feature_extractor.describe(query)
    results = searcher.search(features)

    print(f"INFO: index path: { BASE_DIR / INDEX_FILE }")
    for (result_id, score) in results:
        print(f"INFO: {result_id=}, {score=}")
        matched_photos.append(f"{MEDIA_URL}{result_id}")

    print(f"DEBUG: {matched_photos=}")

