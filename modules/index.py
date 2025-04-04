from modules.feature_extractor import FeatureExtractor
from image_matcher.settings import BASE_DIR

import os
import argparse
import glob

import numpy
import h5py
import cv2

class Indexer:
    def __init__(self) -> None:
        self.feature_extractor = FeatureExtractor()

    def index_images(self, photos_dir: str, index_file_path: str) -> None:
        image_paths = glob.glob(os.path.join(photos_dir, "*.jpg"))
        num_images = len(image_paths)

        feature_dim = self.process_image(image_paths[0]).shape[0]

        with h5py.File(index_file_path, "w") as index_file:
            name_dt = h5py.string_dtype(encoding="utf-8")
            index_file.create_dataset(
                    "image_names",
                    shape=(num_images,),
                    dtype=name_dt
            )

            index_file.create_dataset(
                    "features",
                    shape=(num_images, feature_dim),
                    dtype=numpy.float32
            )

            for idx, image_path in enumerate(image_paths):
                features = self.process_image(image_path)
                image_name = os.path.basename(image_path)
                index_file["features"][idx] = features
                index_file["image_names"][idx] = image_name

    def process_image(self, image_path: str) -> str:
        image_name = image_path.split("/")[-1]

        print(f"INFO: Processing \'{image_name}\'...")

        image = cv2.imread(image_path)
        features = self.feature_extractor.describe(image)

        return features

if __name__ == "__main__":
    indexer = Indexer()

    ap = argparse.ArgumentParser()
    ap.add_argument(
            "-d", "--dataset",
            required = True,
            help = "Path to the directory that contains the images to be indexed"
    )

    index_file = BASE_DIR / "index.hdf5"

    args = vars(ap.parse_args())

    indexer.index_images(args["dataset"], index_file)

