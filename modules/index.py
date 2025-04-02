import concurrent.futures
from typing import List

from modules.colordescriptor import ColorDescriptor

import argparse
import glob
import cv2

class Indexer:
    def __init__(self, color_desc_tup: tuple[int, int, int]) -> None:
        self.color_desc_tup = color_desc_tup

    def index_images(self, photos_dir: str, index_file_path: str) -> None:
        image_paths = glob.glob(photos_dir + "/*.jpg")
        count = 0
        buffer: List[str] = []

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(self.process_image, path)
                for path in image_paths
            ]

            for future in concurrent.futures.as_completed(futures):
                line = future.result()
                buffer.append(line)
                count += 1

        with open(index_file_path, "w") as index_file:
            index_file.writelines(buffer)

        print(f"INFO: Indexed {count} files")

    def process_image(self, image_path: str) -> str:
        color_desc = ColorDescriptor(self.color_desc_tup)
        image_name = image_path.split("/")[-1]

        print(f"INFO: Processing \'{image_name}\'...")
        image = cv2.imread(image_path)

        features = color_desc.describe(image)

        features = [str(f) for f in features]

        return f"{image_name},{','.join(features)}\n"


if __name__ == "__main__":
    indexer = Indexer((8, 12, 3))

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

