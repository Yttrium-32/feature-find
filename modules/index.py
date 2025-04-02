from cv2.typing import MatLike
from modules.colordescriptor import ColorDescriptor
from search_gui.models import ImageFeatures

import argparse
import glob
import cv2
import hashlib

class Indexer:
    def __init__(self, color_desc: ColorDescriptor) -> None:
        self.color_desc = color_desc

    def index_images(self, photos_dir: str) -> None:

        count = 0
        for image_path in glob.glob(photos_dir + "/*.jpg"):
            image_name = image_path.split("/")[-1]

            print(f"INFO: Processing \'{image_name}\'...")
            image = cv2.imread(image_path)

            image_hash = self.hash_image(image)
            if ImageFeatures.objects.filter(image_hash=image_hash).exists():
                print(f"INFO: Skipping duplicate: \'{image_name}\'")
                continue

            features = self.color_desc.describe(image)

            ImageFeatures(
                image_name=image_name,
                image_hash=image_hash,
                features=features
            ).save()

            count += 1

        print(f"INFO: Indexed {count} images")

    def hash_image(self, image: MatLike):
        return hashlib.sha256(image.tobytes()).hexdigest()

if __name__ == "__main__":
    clr_dsc = ColorDescriptor((8, 12, 3))
    indexer = Indexer(clr_dsc)

    ap = argparse.ArgumentParser()
    ap.add_argument(
            "-d", "--dataset",
            required = True,
            help = "Path to the directory that contains the images to be indexed"
    )

    args = vars(ap.parse_args())

    indexer.index_images(args["dataset"])

