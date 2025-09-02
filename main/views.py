import json
from http.client import HTTPResponse
from django.http import HttpResponse, JsonResponse, HttpRequest
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from datasets.views import spam_words, ham_words, predict_message


def prediction_page(request):
    return render(request, 'predictions/home.html')


@csrf_exempt
def english_predictions(request):
    if request.method != "POST":
        return JsonResponse({"status": "Failed", "message": "Only POST requests allowed"}, status=400)
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"status": "Failed", "message": "Invalid JSON"}, status=400)

    input_message = data.get("message")
    if not input_message:
        return JsonResponse({"status": "Failed", "message": "No message provided in request body"}, status=400)

    result = predict_message(input_message, spam_words, ham_words)
    return JsonResponse({"status": "success", "predictions": result})

