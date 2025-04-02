from modules.colordescriptor import ColorDescriptor

import argparse
import glob
import cv2

class Indexer:
    def __init__(self, color_desc: ColorDescriptor) -> None:
        self.color_desc = color_desc

    def index_images(self, photos_dir: str, index_file_path: str) -> None:

        with open(index_file_path, "w") as index_file:
            for image_path in glob.glob(photos_dir + "/*.jpg"):
                image_name = image_path.split("/")[-1]

                print(f"INFO: Processing \'{image_name}\'...")
                image = cv2.imread(image_path)

                features = self.color_desc.describe(image)

                features = [str(f) for f in features]
                index_file.write(f"{image_name},{','.join(features)}\n")

if __name__ == "__main__":
    clr_dsc = ColorDescriptor((8, 12, 3))
    indexer = Indexer(clr_dsc)

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

