from django.shortcuts import render
from django.core.files.storage import default_storage
from .forms import ImageUploadForm
from .classifier import predict_image  # Import function

def home(request):
    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = request.FILES["image"]
            image_path = default_storage.save("uploaded_images/" + image.name, image)

            # ✅ Ensure predict_image() returns two values
            result = predict_image("media/" + image_path)

            # ✅ Check if result contains distance
            if isinstance(result, tuple) and len(result) == 2:
                predicted_label, distance = result
            else:
                predicted_label, distance = result, None  # Handle cases where distance is missing

            return render(request, "result.html", {
                "image_url": "media/" + image_path,
                "label": predicted_label,
                "distance": round(distance, 2) if distance is not None else "N/A"  # Ensure distance is passed
            })

    else:
        form = ImageUploadForm()
    return render(request, "home.html", {"form": form})
