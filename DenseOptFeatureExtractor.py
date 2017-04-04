import cv2
import numpy as np


def DenseOptFlow(cap, Frames):
  ret, frame1 = cap.read()
  prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
  prvs = cv2.resize(prvs, (128,128), interpolation = cv2.INTER_AREA)
  frame1 = cv2.resize(frame1, (128,128), interpolation = cv2.INTER_AREA)
  hsv = np.zeros_like(frame1)
  hsv[:,1] = 255
  count = 0
  while(1):
    count+=1
    ret, frame2 = cap.read()
    if not ret:
      break
    next_ = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    next_ = cv2.resize(next_, (128,128), interpolation = cv2.INTER_AREA)
    flow = cv2.calcOpticalFlowFarneback(prvs,next_, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # import pdb; pdb.set_trace()
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    m = mag.ravel()
    Frames = np.vstack((Frames, m))
    
    cv2.imshow('frame2',bgr)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
      break 
    prvs = next_
  print("count:", count)
  cap.release()
  cv2.destroyAllWindows()
  return (count, Frames)

def main():
  cap = cv2.VideoCapture("videos/boxing.avi")
  DenseOptFlow(cap, Frames)

if __name__=="__main__":
  main()