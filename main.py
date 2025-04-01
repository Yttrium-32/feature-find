from modules.colordescriptor import ColorDescriptor
from modules.searcher import Searcher

import argparse
import cv2

import os
import time

INDEX_PATH: str = "index.csv"

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
            "-q", "--query",
            required=True,
            help="Path to query image"
    )

    ap.add_argument(
            "-p", "--photos",
            required=True,
            help="Path to photos to match"
    )

    args = vars(ap.parse_args())

    clr_dsc = ColorDescriptor((8, 12, 3))

    query = cv2.imread(args["query"])
    features = clr_dsc.describe(query)

    searcher = Searcher(args["photos"], INDEX_PATH, clr_dsc)
    results = searcher.search(features)

    icat = "kitty +kitten icat"
    for (_, result_id) in results:
        os.system(f"{icat} photos/{result_id}")
        time.sleep(1)

if __name__ == "__main__":
    main()
