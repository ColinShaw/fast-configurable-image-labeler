from moviepy.editor   import ImageSequenceClip, load_video, save_video
from src.image_detect import ImageDetect


img_det = ImageDetect()
clip = load_video('input.mp4')

frames = []
for frame in clip.iter_frames():
    frames.append(img_det.detect(frame))

new = ImageSequenceClip(frames, fps = clip.fps)
save_video(new, 'output.mp4');

