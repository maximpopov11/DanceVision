import os

from pytube import YouTube


def download_video():
    YouTube('https://www.youtube.com/watch?v=G5ervgot15Y').streams.first().download(os.getcwd())


if __name__ == '__main__':
    dummy_features = [
        [
            [[1]*33 + [2] * 33],
            [[2]*33 + [1] * 33],
            [[1]*33 + [2] * 33]
        ],
        [
            [[1]*33 + [2] * 33],
            [[1]*33 + [1] * 33],
            [[1]*33 + [2] * 33]
        ],
        [
            [[1]*33 + [2] * 33],
            [[-1]*33 + [1] * 33],
            [[1]*33 + [0] * 33]
        ],
        [
            [[1]*33 + [2] * 33],
            [[1]*33 + [1] * 33],
            [[1]*33 + [0] * 33]
        ],
        [
            [[1]*33 + [2] * 33],
            [[-1]*33 + [1] * 33],
            [[1]*33 + [2] * 33]
        ],
    ]
    # 0 = other
    # 1 = sugar
    # 2 = left
    # 3 = right
    # 4 = whip
    dummy_labels = [0, 1, 2, 3, 4]
