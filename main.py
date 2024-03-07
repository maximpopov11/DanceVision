import os

from pytube import YouTube


def download_video():
    YouTube('https://www.youtube.com/watch?v=G5ervgot15Y').streams.first().download(os.getcwd())


if __name__ == '__main__':
    pass
