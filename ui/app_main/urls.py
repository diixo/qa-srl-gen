
from django.urls import path
from . import views

app_name = "app_main"

urlpatterns = [
    path("", views.main, name="main"),
    path("report", views.report, name="report"),
    path('jobs/', views.jobs, name='jobs'),
    path('jobs/<uuid:pk>/', views.job_detail, name='job'),
    path('jobs/<uuid:pk>/status/', views.job_status, name='job_status'),
    path('jobs/<uuid:pk>/cancel/', views.cancel_job, name='cancel_job'),
    path('jobs/<uuid:pk>/retry/', views.retry_job, name='retry_job'),
    path('jobs/<uuid:pk>/files/<path:name>', views.download, name='download'),
    path('workers/', views.workers, name='workers'),
    path('workers/control/', views.worker_control, name='worker_control'),
    path('corpora/', views.corpora, name='corpora'),
    path('corpora/view/', views.corpus, name='corpus'),
    path('pools/', views.pools, name='pools'),
    path('slots/', views.slots, name='slots'),
    path('upload/', views.upload, name='upload'),
]

urlpatterns += [path(f'{section}/', views.operation, {'section': section}, name=section)
                for section in views.SECTIONS]
