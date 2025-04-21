from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),     
               path("VideoChatbot.html", views.VideoChatbot, name="VideoChatbot"),
	       path("PlaySong", views.PlaySong, name="PlaySong"),	
	       path("StopSound", views.StopSound, name="StopSound"),
	       path("WebCam", views.WebCam, name="WebCam"),	
	       path("DetectEmotion", views.DetectEmotion, name="DetectEmotion"),
]
