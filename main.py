import os

from pytube import YouTube

if __name__ == '__main__':
    YouTube('https://www.youtube.com/watch?v=G5ervgot15Y').streams.first().download(os.getcwd())
