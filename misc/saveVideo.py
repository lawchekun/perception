import cv2
print(cv2.__version__)
vidcap = cv2.VideoCapture('../vis/datasets/GOPR1142.mp4')
success,image = vidcap.read()
count = 0
for _ in range(35):
	while success:
	  cv2.imwrite("frames/frame%d.jpg" % count, image)     # save frame as JPEG file
	  success,image = vidcap.read()
	  print ('Read a new frame: ', success)
	  count += 1
	success = True