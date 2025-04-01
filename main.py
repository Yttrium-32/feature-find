from modules.colordescriptor import ColorDescriptor
from modules.searcher import Searcher
from image_matcher.settings import STATIC_URL

import argparse
import cv2

import os
import time

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
            "-q", "--query",
            required=True,
            help="Path to query image"
    )

    args = vars(ap.parse_args())

    clr_dsc = ColorDescriptor((8, 12, 3))

    query = cv2.imread(args["query"])
    features = clr_dsc.describe(query)

    searcher = Searcher(STATIC_URL, STATIC_URL + "index.csv", clr_dsc)
    results = searcher.search(features)

    icat = "kitty +kitten icat"
    for (_, result_id) in results:
        os.system(f"{icat} {STATIC_URL}{result_id}")
        time.sleep(1)

if __name__ == "__main__":
    main()
