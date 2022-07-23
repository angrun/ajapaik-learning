import json

from app.service import TrainigService
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt


def get_prediction(request, image_url):
    print(image_url)
    if request.method == 'GET':
        image_url = image_url.replace("-", "/").replace("/home/docker/ajapaik/", "/home/anna/ajapaik-web/")
        response = TrainigService().predict(image_url)
        return HttpResponse(json.dumps(response), content_type="application/json")

@csrf_exempt
def post_prediction(request):
    if request.method == 'POST':
        print("Hello")
        return HttpResponse(json.dumps("{status:200}"), content_type="application/json")
