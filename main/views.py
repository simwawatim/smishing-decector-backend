import json
from http.client import HTTPResponse
from django.http import HttpResponse, JsonResponse, HttpRequest
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from datasets.views import spam_words, ham_words, predict_message
from rest_framework import viewsets
from datasets.models import SMSMessage
from .serializers import SMSMessageSerializer


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



# ------------------- CREATE -------------------
@csrf_exempt
def create_sms(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            label = data.get("label")
            message = data.get("message")

            if not label or not message:
                return JsonResponse({"status": "Failed", "message": "Label and message are required"}, status=400)

            sms = SMSMessage.objects.create(label=label, message=message)
            return JsonResponse({
                "status": "Success",
                "id": sms.id,
                "label": sms.label,
                "message": sms.message,
                "created_at": sms.created_at
            })
        except json.JSONDecodeError:
            return JsonResponse({"status": "Failed", "message": "Invalid JSON"}, status=400)
    else:
        return JsonResponse({"status": "Failed", "message": "Only POST allowed"}, status=405)


# ------------------- READ ALL -------------------
def read_all_sms(request):
    if request.method == "GET":
        messages = SMSMessage.objects.all().order_by('-created_at')
        data = [
            {
                "id": msg.id,
                "label": msg.label,
                "message": msg.message,
                "created_at": msg.created_at
            }
            for msg in messages
        ]
        return JsonResponse({"status": "Success", "data": data})
    return JsonResponse({"status": "Failed", "message": "Only GET allowed"}, status=405)


# ------------------- READ SINGLE -------------------
def read_one_sms(request, pk):
    if request.method == "GET":
        try:
            msg = SMSMessage.objects.get(pk=pk)
            data = {
                "id": msg.id,
                "label": msg.label,
                "message": msg.message,
                "created_at": msg.created_at
            }
            return JsonResponse({"status": "Success", "data": data})
        except SMSMessage.DoesNotExist:
            return JsonResponse({"status": "Failed", "message": "Message not found"}, status=404)
    return JsonResponse({"status": "Failed", "message": "Only GET allowed"}, status=405)


# ------------------- UPDATE -------------------
@csrf_exempt
def update_sms(request, pk):
    if request.method == "PUT":
        try:
            msg = SMSMessage.objects.get(pk=pk)
            data = json.loads(request.body)
            msg.label = data.get("label", msg.label)
            msg.message = data.get("message", msg.message)
            msg.save()
            return JsonResponse({"status": "Success", "message": "Updated successfully"})
        except SMSMessage.DoesNotExist:
            return JsonResponse({"status": "Failed", "message": "Message not found"}, status=404)
        except json.JSONDecodeError:
            return JsonResponse({"status": "Failed", "message": "Invalid JSON"}, status=400)
    return JsonResponse({"status": "Failed", "message": "Only PUT allowed"}, status=405)


@csrf_exempt
def delete_sms(request, pk):
    if request.method == "DELETE":
        try:
            msg = SMSMessage.objects.get(pk=pk)
            msg.delete()
            return JsonResponse({"status": "Success", "message": "Deleted successfully"})
        except SMSMessage.DoesNotExist:
            return JsonResponse({"status": "Failed", "message": "Message not found"}, status=404)
    return JsonResponse({"status": "Failed", "message": "Only DELETE allowed"}, status=405)
