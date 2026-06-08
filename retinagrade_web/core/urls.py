from django.urls import path
from . import views

app_name = 'core'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('analyse/', views.analyse, name='analyse'),
    path('history/', views.history, name='history'),
    path('prediction/<int:pk>/', views.prediction_detail, name='prediction_detail'),
    path('prediction/<int:pk>/notes/', views.save_notes, name='save_notes'),
    path('prediction/<int:pk>/delete/', views.delete_prediction, name='delete_prediction'),
    path('prediction/<int:pk>/report/', views.download_report, name='download_report'),
    path('history/bulk-delete/', views.bulk_delete, name='bulk_delete'),
    path('admin-dashboard/', views.admin_dashboard, name='admin_dashboard'),
]
