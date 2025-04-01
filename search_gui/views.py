import cv2

from django.shortcuts import render
from django.http import HttpRequest
from django.core.files.storage import FileSystemStorage

from modules.searcher import Searcher
from modules.colordescriptor import ColorDescriptor

from image_matcher.settings import BASE_DIR, MEDIA_ROOT, MEDIA_URL

def search_gui(request: HttpRequest):
    uploaded_image = None
    matched_photos = []

    if request.method == "POST" and request.FILES.get("image"):
        image = request.FILES["image"]
        fs = FileSystemStorage(location=MEDIA_ROOT / 'user_uploads')
        filename = fs.save(image.name, image)
        uploaded_image = MEDIA_URL + 'user_uploads/' + filename

        saved_image_path = fs.path(filename)
        query = cv2.imread(saved_image_path)

        clr_dsc = ColorDescriptor((8, 12, 3))
        searcher = Searcher(
                MEDIA_ROOT.__str__(),
                BASE_DIR / "index.csv",
                clr_dsc
        )

        features = clr_dsc.describe(query)
        results = searcher.search(features)

        print(f"DEBUG: {BASE_DIR / 'index.csv'}")
        print(f"DEBUG: {results=}")
        for (_, result_id) in results:
            print(f"DEBUG: {result_id=}")
            matched_photos.append(f"{MEDIA_URL}{result_id}")

        print(f"DEBUG: {matched_photos=}")

    return render(request, 'index.html', {
        'uploaded_image_url': uploaded_image,
        'result_images': matched_photos
    })
