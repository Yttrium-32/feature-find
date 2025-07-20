from feature_find.settings import BASE_DIR, INDEX_FILE, MEDIA_ROOT, MEDIA_URL
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

    def compute_cos_sim(self, image_idx, image_features, query_features):
        vec_image = numpy.array(image_features).reshape(1, -1)
        vec_query = numpy.array(query_features).reshape(1, -1)

        sim = cosine_similarity(vec_image, vec_query)[0][0]

        return (sim, image_idx)

    def match_label(self, image_label, query_label):
        return image_label == query_label

    def search(self, query_features, query_label, limit: int = 10):
        results = []

        with h5py.File(self.index_path, "r") as index_file:
            all_features = index_file["features"][:]
            image_names = index_file["image_names"][:]
            image_labels = index_file["labels"][:]

            # Extract images with similar features
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        self.compute_cos_sim,
                        idx,
                        all_features[idx],
                        query_features
                    )
                    for idx in range(all_features.shape[0])
                ]
                for future in as_completed(futures):
                    sim, image_idx = future.result()
                    if sim > 0.5:
                        results.append({
                            "image_name": image_names[image_idx].decode(),
                            "similarity": sim,
                            "features": all_features[image_idx],
                            "label": image_labels[image_idx].decode()
                        })

        results.sort(key=lambda item: item["similarity"], reverse=True)

        # Furthur refine search by matching labels
        with ThreadPoolExecutor() as executor:
            matches = executor.map(
                    lambda item: self.match_label(item["label"], query_label),
                    results
            )

        refined_results = []
        for match, item in zip(matches, results):
            if match:
                image_name = item["image_name"]
                image_sim = item["similarity"]
                image_label = item["label"]
                refined_results.append((image_name, image_sim, image_label))

        return refined_results[:limit]

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

    features, label = feature_extractor.describe(query)
    results = searcher.search(features, label)

    print(f"INFO: index path: { BASE_DIR / INDEX_FILE }")
    for (result_id, score) in results:
        print(f"INFO: {result_id=}, {score=}")
        matched_photos.append(f"{MEDIA_URL}{result_id}")

    print(f"INFO: Extracted label: {label}")
    print(f"DEBUG: {matched_photos=}")

