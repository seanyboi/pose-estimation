# STEP 1
Load the Caffe Network - .prototxt contains the architecture of the neural network. In this case the neural network consists of 10 layers of VGGNET and then a 2-branch multi-stage CNN. This architecture won the COCO keypoints challenge in 2016 by the Perceptual Lab at Carnegie Mellon University - .caffemodel file stores the weights of the trained model.

# STEP 2
Read the image and prepare input into the network and then make predictions and parse the key points. Key points are identified if confidence is higher than threshold.

# STEP 3
Place circles onto image showing key points.

# STEP 4
Draw skeleton connecting key points.

Ouput of key points can be seen in 'Output-Keypoints.jpg' and output of skeleton can be seen in 'Output-Skeleton.jpg'
