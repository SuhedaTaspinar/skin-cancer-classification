import os
from django.conf import settings
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.utils.text import slugify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Eğitilmiş modeli yükleyin
model = load_model(os.path.join(settings.BASE_DIR, 'modeller/optimized_skin_cancer_model.h5'))

# Cilt hastalığı sınıfları
cilt_turleri = ["Melanom", "Nevüs", "Benign Keratoz", "Bazal Hücreli Karsinom", "Aktinik Keratoz", "Deri Yaması",
                "Karsinom"]


def tanila(request):
    if request.method == 'POST' and request.FILES['image']:
        # Dosya adını temizlemek için slugify kullanıyoruz
        image = request.FILES['image']
        file_name = slugify(image.name)  # Dosya adını güvenli hale getiriyoruz

        # Medya klasörüne dosyayı kaydediyoruz
        file_path = default_storage.save(file_name, ContentFile(image.read()))
        image_path = os.path.join(settings.MEDIA_ROOT, file_path)

        # Resmi açın, boyutlandırın ve model için hazırlayın
        img = Image.open(image_path)
        img = img.resize((100, 75))
        img_array = np.array(img) / 255.0  # Normalizasyon
        img_array = np.expand_dims(img_array, axis=0)

        # Model ile tahmin yapın
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        tahmin = cilt_turleri[class_index]

        # Resmi göstermek için image_url'i MEDIA_URL ile birleştirerek oluşturun
        image_url = settings.MEDIA_URL + file_path

        return render(request, 'sonuc.html', {'tahmin': tahmin, 'image_url': image_url})

    return render(request, 'index.html')
