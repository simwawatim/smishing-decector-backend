from detector.urls import path
from .views import (
    prediction_page,
    english_predictions
)

app_name = 'action'
urlpatterns = [
    path('',  prediction_page, name='prediction_page'),
    path('english-predidctions/', english_predictions, name='english_predidctions'),
]
