from moviepy.editor   import ImageSequenceClip, load_video, save_video
from src.image_detect import ImageDetect
import sys


img_det = ImageDetect()
clip = load_video(argv[0])

frames = []
for frame in clip.iter_frames():
    frames.append(img_det.label_image(frame))

new = ImageSequenceClip(frames, fps = clip.fps)
save_video(new, argv[1]);

