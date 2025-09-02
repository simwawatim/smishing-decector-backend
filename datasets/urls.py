from detector.urls import path

from . import views

app_name = "datasets"
urlpatterns = [

    path('', views.home, name='home'),
    path('messages-list/', views.messages_list, name='messages_list'),
    path('update/', views.update_data_sets, name='update_data_sets'),
    path('create-dataset/', views.create_datasets, name='create_datasets'),

]