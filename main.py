import cv2
import sys
import numpy as np
import imutils

if __name__ == '__main__':

    tracker = cv2.TrackerMIL_create()
    video = cv2.VideoCapture("pilkarzyki.mp4")
    ok, frame = video.read()
    frame = imutils.resize(frame, width=1000)
    orangeLower = (0,50,150)
    orangeUpper = (50,180,200)
    boundaries = [([0, 50, 150], [50, 180, 200])]

    #frame = imutils.resize(frame, width=600)
    # hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    # mask = cv2.inRange(hsv,orangeLower,orangeUpper)
    # mask = cv2.erode(mask,None,iterations = 2)
    # mask = cv2.dilate(mask,None,iterations = 2)
    # loop over the boundaries
    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.erode(mask,None,iterations = 2)
        mask = cv2.dilate(mask,None,iterations = 2)
        #output = cv2.bitwise_and(frame, frame, mask=mask)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    #bbox = (int(x), int(y), radius, radius)

    #ok = tracker.init(frame, bbox)

    while True:

        (ok, frame) = video.read()
        frame = imutils.resize(frame, width=1000)
        if not ok:
            break
        for (lower, upper) in boundaries:
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
            #hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(frame, lower, upper)
            #mask = cv2.erode(mask,None,iterations = 2)
            #mask = cv2.dilate(mask,None,iterations = 2)
            #output = cv2.bitwise_and(frame, frame, mask=mask)
        timer = cv2.getTickCount()
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            #M = cv2.moments(c)
            #center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # only proceed if the radius meets a minimum size
            if radius > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                #cv2.circle(frame, center, 5, (0, 0, 255), -1)
        #bbox = (int(x), int(y), radius, radius)
        #ok, bbox = tracker.update(frame)

        #fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)



        cv2.imshow("Tracking", frame)

        k = cv2.waitKey(1) & 0xff
        if k == 27: break

