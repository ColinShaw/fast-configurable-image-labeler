These are some scripts that can be helpful in generating the cat
face sub-images from the Oxford-IIIT pet data set, as well as
generating class images for training from a video.  

For the Oxford-IIIT data set, you can drop the `/images/` directory
from the data set into this directory and just run the
`make_oxford_iiit_faces.py` script.  This will produce the 
facial cropped entries in the `/data/negative/`
directory.  The file `cats.csv` is a list of the file names
of only the cats in the data set.  The data set itself can be
obtained from `http://www.robots.ox.ac.uk/~vgg/data/pets/`.  While
this gives you a good start, it isn't perfect, so you may
want to review the results, though probably unnecassary for the
negative set.

For the specific positive image classes, you can simply 
run the `video_to_class_samples.py` script with an argument
of an image name.  This will generate class examples
in the `/data/positive/` directory indexed by the 
class name.  You probably want to review the detected images
and discard those that are not of good quality.

