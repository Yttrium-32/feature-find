from modules.colordescriptor import ColorDescriptor
import argparse
import glob
import cv2

ap = argparse.ArgumentParser()
ap.add_argument(
        "-d", "--dataset",
        required=True,
        help = "Path to the directory that contains the images to be indexed"
)

ap.add_argument(
        "-i", "--index",
        required=True,
        help = "Path to where the computed index will be stored"
)

args = vars(ap.parse_args())

cd = ColorDescriptor((8, 12, 3))

with open(args["index"], "w") as index_file:
    for image_path in glob.glob(args["dataset"] + "/*.jpg"):
        image_name = image_path.split("/")[-1]
        image = cv2.imread(image_path)

        features = cd.describe(image)

        features = [str(f) for f in features]
        index_file.write(f"{image_name},{','.join(features)}")
