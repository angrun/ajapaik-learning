from django.http import HttpResponse

from app.service import TrainigService
import json


def get_prediction(request, image_url):
    print(image_url)
    if request.method == 'GET':
        response_data = {'probabilities': [0.91004425, 0.0899557]}
        image_url = image_url.replace("-", "/").replace("/home/docker/ajapaik/", "/home/anna/ajapaik-web/" )
        print(image_url)
        response = TrainigService().predict(image_url)
        print(response)
        return HttpResponse(json.dumps({"probability": "0.91004425", "category": "interior"}), content_type="application/json")