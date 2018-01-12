import argparse


p = argparse.ArgumentParser()

p.add_argument('-v', '--video',    action='append', nargs=2, help='detect from a video')
p.add_argument('-r', '--realtime', action='store_true',      help='detect from live camera feed')
p.add_argument('-t', '--test',     action='store_true',      help='test camera feed')

args = p.parse_args()

if args.video:
    from src.video_detect import VideoDetect
    source = args.video[0][0]
    destination = args.video[0][1]
    VideoDetect().detect(source, destination)
elif args.realtime:
    from src.realtime_detect import RealtimeDetect
    RealtimeDetect().detect()
elif args.test:
    from src.realtime_test import RealtimeTest
    RealtimeTest().test()
else:
    p.print_help()

