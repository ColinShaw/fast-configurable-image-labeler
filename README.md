# Fast Configurable Image Labeler

This is a tool for quickly training a classifier
to label specific examples of common classes
of images.  The goal is minimal training time
with good predictive power, and an interface
that makes the task simple to tailor to new 
labeling problems.  The original purpose of this
was to create a tool for realtime identification
of cats in video with minimal training required
to update the cats that would be tracked.



### Technique

Labeling images with specific class names is 
not one where you want to train a complete 
network, as the cost of backprobagation with
enough training examples is quite high.  Not 
even freezing part of your network, as the part
that you are going to be training is likely 
dense layers, which have (go figure) the highest
weight density.  From a cost perspective, this
rules out adding new top ends and things like
siamese networks despite having fairly simple
top ends.  Moreover, you can't add new 
classes all that easily without completely 
reconsidering the part of the network that you 
are training taking the approach of having to
train the actual network.  You need to pull the
classification task out of the network to 
facilitate fast training.

What is commonly done is take the image data 
and find a small basis space with maximum 
span that can be used with a space partitioning
classifier.  What is done here is use the 
convolutional part of MobileNet to produce 
a vector that can be cast more easily to a
smaller support space than the original 
image.  MobileNet is, of course, a classifier, 
so what we are doing is taking advantage of 
those aspects of that training that give us 
convolutional feature weights, and we are 
disregarding the dense classifier top end.  Why
MobileNet specifically?  Size.  Part of the 
objective is to be able to run the detector
on small hardware in as close to realtime as
possible.

In the case of MobileNet, prior to the dense
classification section, we are left with a 
vector space of --- dimensions.  This is an
improvement, as the original scaled images feeding
the network span a space of 49,152 dimensions, but 
the real improvement is in the degree that we can 
project this down further.  The benefit of using 
the convnet first is that the vector before the 
classifier has effectively been trained to be good 
at reducing the space.  Of course there are many
other techniques for doing similar, such as using
a histogram of oriented gradients approach.  However,
the HOG approach doesn't abstract to examples that
are not trained as readily.  Because of this, 
it tends to need a larger spanning space for the
classifier, which results in a more complex 
top end.  Using something like MobileNet simply
gives better performance in this situation.

The output of the convolutional layers of 
MobileNet, when subjected to the training data,
are used to train a pricipal components 
analysis model with restricted dimensional 
output.  If you experiment a bit with MobileNet,
you will find that it actually doesn't emit the
full cadre of --- dimensions for particular
classes, but rather something a bit smaller.  For 
cat data, that something is only 3,341 significant
dimensions.  That is great, already an improvement
of nearly 15:1 from the original image data.  However,
limiting the result of PCA to dimensionality in the 
10 to 50 range works much better.  What that means
is that by making intelligent transformations on
our original image data, we are able to effectively
describe the differences between different 
objects in the same class with less than a 
thousandth the original data.  The specific 
selection can be optimized by performing validation
on the detection and looking at the magnitude of the 
descriptive power being discarded by the approach.  

Now that the dimensionality is under control, and 
we have a space that encapsulates much of the 
essence of the image, a conventional classifier is
used.  What proves to work well is a linear 
support vector machine classifier.  Other kernels
tend to overfit and cause images to be 
misdetected.  The result of using PCA on the 
convolutional outputs in practice work conveniently
well with the linear kernel.

The initial image is detected using MobileNet
trained as a single shot detector.  Again, the reason
for using MobileNet is the size of the network as
relates to using it on resource constrained 
hardware.  You can read more about this approach
[here](https://github.com/weiliu89/caffe/tree/ssd) 
and [here](https://github.com/chuanqi305/MobileNet-SSD).



### Workflow

This application basically dispatches a number of
somewhat unrelated features.  These generally fall
under the categories of generating data, training
and using data, and cleaning up old models.

The simplest way to generate training data, at least
in the context of what this was initially intended
for, is from video.  First, open `config.yaml` and
set the class to what you want to detect.  Most
common would be either `cat` or `person`, though 
there are a number of other classes listed in 
`/src/classes.py` that can be used.  You also need
to set the confidence threshold for detections of 
the specified class.

First, get in the habit of clearing out old 
models prior to using new data:
```
python app.py -d
```

Next record some videos of what you intend to train
to.  Put these in the `/videos/` directory.  The 
naming convention is to make the names of the form
`<class><#>.mp4`, where class is the class name you
want and the suffix is some number to disambiguate
multiple videos for a particular class.  When you
have all of the videos recorded, simply run:
```
python app.py -g
```

This will go through the videos and generate a 
bunch of detected `.png`s that live in the 
`/data/` directory under directories with the 
same name as the class.  You should go through 
these images after they are detected to make sure 
they seem appropriate.  Features that are 
completely wrong or that are particularly specific
aspects of the object may not be the best to train
with.  

To make it simpler for generating the videos, you
can use the built-in video feature.  Invoke it
like this:
```
python app.py -v <videoname>
```

This will launch a preview window with video from
your webcam in it, and upon pressing the `q` key
from the terminal you launched it from, it will
save the video in the `/videos/` directory with
the name specified.  Of course the name needs to
comply with the convention of a class name and
a disambiguation number as noted earlier.

You will need to create a `null` class as well,
something that is representative of not being in 
the detection class.  This is an absolute 
requirement if there is only one class, as otherwise
there is no way for the classifier to split the
classes.  It is also a good idea regardless of the
other classes.  The specific name of this class is
defined in `config.yaml` and relates to the name
of the folder with images in it in the `/data/`
directory.  Since the video generator of these
images is based on the specific detection class, it
won't be of use of the `null` class.  You probably 
don't need or want many examples.  Things that would
be good would be the background from the video 
frame, any aberrant background items (e.g. if there
are cars in a street in the background), etc.  Just
capture a few shots and dumpy them in the `/data/` 
directory under the `null` class.

At this point you can do two things, either process
from a live video or create a labeled video from 
an unlabeled video.  In either case, the final models
are trained the first time this is engaged (e.g. PCA
and linear SVC).  The two ways to invoke this are:
```
python app.py -r
python app.py -v <input> <output>
```



### Results

