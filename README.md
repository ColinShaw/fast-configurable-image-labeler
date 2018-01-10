# Fast Cat Face Labeler



### Models and data

The LBP and HOG cascades are those from `vision-ary` (now `ARGO Vision`) that can 
be found here:

```
http://www.vision-ary.net/2015/03/boost-the-world-cat-faces/
```

The VGG16 model is the stock pre-trained Keras model trained on ImageNet data
with `include_top=False`.

Cat training data for `null` class cats is from the Oxford-IIIT pet data
set.  The cat images have the cat faces extracted.

