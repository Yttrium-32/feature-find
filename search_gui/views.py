import cv2

from django.shortcuts import render
from django.http import HttpRequest
from django.core.files.storage import FileSystemStorage

from modules.searcher import Searcher
from modules.feature_extractor import FeatureExtractor

from image_matcher.settings import INDEX_FILE, MEDIA_ROOT, MEDIA_URL

feature_extractor = FeatureExtractor()

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

        searcher = Searcher(
            MEDIA_ROOT.__str__(),
            INDEX_FILE,
        )

        features, label = feature_extractor.describe(query)
        results = searcher.search(features, label)

        print(f"DEBUG: { INDEX_FILE }")
        for result_id, result_sim, result_label in results:
            print(f"DEBUG: {result_id=}, {result_sim=}, {result_label=}")
            matched_photos.append(f"{MEDIA_URL}{result_id}")

        print(f"DEBUG: {matched_photos=}")
        print(f"DEBUG: Extracted label: { label }")

    return render(request, 'index.html', {
        'uploaded_image_url': uploaded_image,
        'result_images': matched_photos
    })

