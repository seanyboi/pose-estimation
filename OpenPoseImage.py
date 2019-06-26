import cv2
import time
import numpy as np

# MODE = MPI
# Specify the paths for the 2 files

MODE = "COCO"

if MODE is "COCO":
    protoFile = "poseData/coco/pose_deploy_linevec.prototxt"
    weightsFile = "poseData/coco/pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]
elif MODE is "MPI":
    protoFile = "poseData/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "poseData/mpi/pose_iter_160000.caffemodel"
    nPoints = 15
    POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]

# Read image
frame = cv2.imread("person.jpeg")
frameCopy = np.copy(frame)
frameWidth = frame.shape[1]
frameHeight = frame.shape[0]
threshold = 0.1

# Read the network into Memory
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

t = time.time()
# Specify the input image dimensions
inWidth = 368
inHeight = 368

# Prepare the frame to be fed to the network (so Caffe can deal with it)
# 1. Normalize
# 2. Specify dimensions
# 3. Mean value to be subtracted
# no need to swap channels as both use RGB
inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
# Set the prepared object as the input blob of the network
net.setInput(inpBlob)

# Make predictions
# 4-D Matrix
# First Dimension - Image ID .
# Second Dimension - Index of a keypoint.
# Third Dimension - Height of the output.
# Fourth Dimension - Width of the output.
output = net.forward()
print("time taken by network : {:.3f}".format(time.time() - t))

# Recieve location of keypoint by finding maxima of confidence map of that keyopoint
# Plot keypoints on image

H = output.shape[2]
W = output.shape[3]

# Empty list to store the detected points
points = []
for i in range(nPoints):
    # Confidence map of corresponding body parts
    probMap = output[0, i, :, :]

    # Find global maxima of the probMap (confidence score)
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

    # Scale the point to fit on the original Image
    x = (frameWidth * point[0]) / W
    y = (frameHeight * point[1]) / H

    if prob > threshold :
        cv2.circle(frameCopy, (int(x), int(y)), 15, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, lineType=cv2.LINE_AA)

        # Add the point to the list if the probability is greater than the threshold
        points.append((int(x), int(y)))
    else:
        points.append(None)

# Draw Skeleton
for pair in POSE_PAIRS:
    partA = pair[0]
    partB = pair[1]

    if points[partA] and points[partB]:
        cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
        cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)


cv2.imshow('Output-Keypoints', frameCopy)
cv2.imshow('Output-Skeleton', frame)
cv2.imwrite('Output-Keypoints.jpg', frameCopy)
cv2.imwrite('Output-Skeleton.jpg', frame)
print("Total time taken : {:.3f}".format(time.time() - t))
cv2.waitKey(0)
cv2.destroyAllWindows()
