These are some scripts that can be helpful in generating the cat
face sub-images from the Oxford-IIIT pet data set, as well as
generating class images for training from a video.  

For the Oxford-IIIT data set, you can drop the `/images/` directory
from the data set into this directory and just run the
`make_oxford_iiit_faces.py` script.  This will produce the 
facial cropped entries in the `/data/negative/`
directory.  This script actually applies the cat face detector
to all of the images, both cats and dogs.  This results in 
some images of dog faces (and some just bad detections) in the
negative feature set.  This isn't a big deal since it is the 
negative feature set and we don't want to detect the dogs or 
the error detections as member of our positive image classes
anyway.  The Oxford-IIIT pet data set itself can be obtained 
[here](http://www.robots.ox.ac.uk/~vgg/data/pets/). 

For the specific positive image classes, you can simply 
run the `video_to_class_samples.py` script with an argument
of a video file name.  This will generate class examples
in the `/data/positive/` directory indexed by the 
class name from frames in the video.  You probably want to 
review the detected images and discard those that are not 
correct label or are not of good quality since we do not 
want to false-positive these.

