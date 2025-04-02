from modules.colordescriptor import ColorDescriptor
from modules.searcher import Searcher
from image_matcher.settings import BASE_DIR, MEDIA_ROOT

import argparse
import cv2

import time

INDEX_PATH: str = (BASE_DIR / "index.csv").__str__()
PHOTO_PATH: str = MEDIA_ROOT.__str__()

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

    print(f"{PHOTO_PATH=}")
    searcher = Searcher(
            PHOTO_PATH,
            INDEX_PATH,
            clr_dsc
    )
    results = searcher.search(features)

    for (result_id, dist) in results:
        print(f"{PHOTO_PATH}/{result_id}: {dist}")
        time.sleep(1)

if __name__ == "__main__":
    main()

