import json

from app.service import TrainigService
from django.http import HttpResponse


def get_prediction(request, image_url):
    print(image_url)
    if request.method == 'GET':
        image_url = image_url.replace("-", "/").replace("/home/docker/ajapaik/", "/home/anna/ajapaik-web/")
        response = TrainigService().predict(image_url)
        return HttpResponse(json.dumps(response), content_type="application/json")


def post_prediction(request):
    if request.method == 'POST':
        print("Hello")
