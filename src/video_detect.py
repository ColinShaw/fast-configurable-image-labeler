from moviepy.editor import ImageSequenceClip, VideoFileClip
from image_detect   import ImageDetect


class VideoDetect(object):

    def __init__(self):
        self.__imgdet = ImageDetect()

    def detect(self, source, destination):
        clip, frames = VideoFileClip(source), []
        for frame in clip.iter_frames():
            frame = self.__imgdet.label_image(frame)
            frames.append(frame)
        clip = ImageSequenceClip(frames, fps=clip.fps)
        clip.write_videofile(destination, audio=False)

