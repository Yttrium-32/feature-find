from modules.feature_extractor import FeatureExtractor

import argparse
import glob
import cv2

class Indexer:
    def __init__(self) -> None:
        self.feature_extractor = FeatureExtractor()

    def index_images(self, photos_dir: str, index_file_path: str) -> None:
        with open(index_file_path, "w") as index_file:
            for image_path in glob.glob(photos_dir + "/*.jpg"):
                feature_arr = self.process_image(image_path)
                index_file.write(feature_arr)

    def process_image(self, image_path: str) -> str:
        image_name = image_path.split("/")[-1]

        print(f"INFO: Processing \'{image_name}\'...")

        image = cv2.imread(image_path)
        features = self.feature_extractor.describe(image)

        features = [str(f) for f in features]

        return f"{image_name},{','.join(features)}\n"

if __name__ == "__main__":
    indexer = Indexer()

    ap = argparse.ArgumentParser()
    ap.add_argument(
            "-d", "--dataset",
            required = True,
            help = "Path to the directory that contains the images to be indexed"
    )
    ap.add_argument(
            "-i", "--index",
            required = True,
            help = "Path to where the computed index will be stored"
    )

    args = vars(ap.parse_args())

    indexer.index_images(args["dataset"], args["index"])

