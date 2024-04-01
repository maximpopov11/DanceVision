import os

from pytube import YouTube


def download_videos():
    YouTube('https://www.youtube.com/watch?v=G5ervgot15Y').streams.first().download(os.getcwd())


download_videos()
