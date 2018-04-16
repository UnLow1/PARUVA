################################
####### OBJECT DETECTION #######
################################
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# img_rgb = cv2.imread('mario.png')
# img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
# template = cv2.imread('mario_coin.png', 0)
# w, h = template.shape[::-1]
#
# res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
# threshold = 0.8
# loc = np.where(res >= threshold)
# for pt in zip(*loc[::-1]):
#     cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
#
# cv2.imwrite('res.png', img_rgb)

################################
###### VIDEO FROM CAMERA #######
################################
# import numpy as np
# import cv2
#
# cap = cv2.VideoCapture(0)
#
# while (True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#
#     # Our operations on the frame come here
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Display the resulting frame
#     cv2.imshow('frame', gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()



################################
###### VIDEO FROM FILE #########
################################
import numpy as np
import cv2

cap = cv2.VideoCapture('pilkarzyki.mp4')
template = cv2.imread('pilka.png', 0)
w, h = template.shape[::-1]

boundaries = [([255, 102, 255], [153, 51, 255])]

while cap.isOpened():
    ret, frame = cap.read()

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    # res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    # threshold = 0.7
    # loc = np.where(res >= threshold)
    # for pt in zip(*loc[::-1]):
    #     cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    #
    # # cv2.imwrite('res.png', img_rgb)
    #
    # cv2.imshow('frame', frame)
    #
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break


    # loop over the boundaries
    for (lower, upper) in boundaries:
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(frame, lower, upper)
        output = cv2.bitwise_and(frame, frame, mask=mask)

        # show the images
        cv2.imshow("images", np.hstack([frame, output]))
        # cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
