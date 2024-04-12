import os

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from pytube import YouTube

videos = [
    "https://www.youtube.com/watch?v=G5ervgot15Y"
]

# Each clip from second marker at one index to the next represents a single movement in the respective video
clips = [
    [61, 76, 79.5, 83.5, 86, 89, 92.5, 94.5, 97, 100, 103, 106, 108, 110, 111.5, 115.5, 118, 120, 122.5, 126, 129.5, 132,
     135, 138, 143, 145.5, 148, 152, 156, 158, 161.5, 164.5, 166.75, 170.25, 173.5, 176.5, 183, 187, 190]
]

# 0 = sugar push/tuck
# 1 = left side pass
# 2 = right side pass
# 3 = whip
# 4 = other
labels = [
    [4, 1, 0, 1, 1, 4, 0, 2, 0, 3, 3, 0, 1, 2, 4, 0, 0, 4, 1, 1, 0, 1, 1, 3, 4, 0, 0, 2, 1, 3, 0, 1, 0, 0, 2, 3, 4, 0]
]


def download_videos():
    for i, video in enumerate(videos):
        YouTube(video).streams.first().download(os.getcwd(), filename=f"v{i}.mp4")


def clip_videos():
    for i in range(len(clips)):
        for j in range(len(clips[i]) - 1):
            ffmpeg_extract_subclip(f"v{i}.mp4", clips[i][j], clips[i][j+1], targetname=f"v{i}_{j}.mp4")


download_videos()
clip_videos()
