################################
####### OBJECT DETECTION #######
################################
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
###### CAPTURE FROM TEMPLATE #######
################################
# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
# res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
# threshold = 0.8
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


################################
###### VIDEO FROM FILE #########
################################
import numpy as np
import cv2

set_colors = False
detect_ball = True
detect_red_players = False
detect_blue_players = False
# template: boundaries = [([G_min, B_min, R_min], [G_max, B_max, R_max])]
boundaries = [([0, 50, 150], [50, 180, 200])] #[([0, 0, 190], [62, 174, 250])]
boundaries_ball = [([0, 50, 150], [50, 180, 200])]
boundaries_red_players = [([0, 0, 140], [80, 80, 255])]
boundaries_blue_players = [([100, 0, 0], [255, 100, 50])]
filename = 'pilkarzyki.mp4'


def main():
    cap = cv2.VideoCapture(filename)
    # cap = cv2.VideoCapture('example.jpg')
    # template = cv2.imread('pilka.png', 0)
    # w, h = template.shape[::-1]



    while cap.isOpened():
        boundaries = set_proper_boudnaries()

        ret, frame = cap.read()

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
            cv2.imshow("images", output)

            if set_colors:
                cv2.waitKey(0)
                boundaries = set_colors(boundaries)
            else:
                cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()


def set_proper_boudnaries():
    boundaries = [([25, 146, 190], [62, 174, 250])]
    if detect_ball:
        boundaries = boundaries_ball
    elif detect_red_players:
        boundaries = boundaries_red_players
    elif detect_blue_players:
        boundaries = boundaries_blue_players
    return boundaries


def set_colors(boundaries):
    r_min = boundaries[0][0][2]
    r_max = boundaries[0][1][2]
    g_min = boundaries[0][0][0]
    g_max = boundaries[0][1][0]
    b_min = boundaries[0][0][1]
    b_max = boundaries[0][1][1]

    print("=================================")
    print("R: " + str(r_min) + " - " + str(r_max))
    print("G: " + str(g_min) + " - " + str(g_max))
    print("B: " + str(b_min) + " - " + str(b_max))
    print("=================================")

    key = cv2.waitKey(33)
    # red colour
    if key == ord('R') and r_max < 255:
        r_max += 1
    elif key == ord('r') and r_max > 0:
        r_max -= 1
    elif key == ord('E') and r_min < 255:
        r_min += 1
    elif key == ord('e') and r_min > 0:
        r_min -= 1
    # green colour
    elif key == ord('G') and g_max < 255:
        g_max += 1
    elif key == ord('g') and g_max > 0:
        g_max -= 1
    elif key == ord('F') and g_min < 255:
        g_min += 1
    elif key == ord('f') and g_min > 0:
        g_min -= 1
    # blue colour
    elif key == ord('B') and b_max < 255:
        b_max += 1
    elif key == ord('b') and b_max > 0:
        b_max -= 1
    elif key == ord('V') and b_min < 255:
        b_min += 1
    elif key == ord('v') and b_min > 0:
        b_min -= 1
    elif key == ord('h'):
        print("=================================")
        print("R: " + str(r_min) + " - " + str(r_max))
        print("G: " + str(g_min) + " - " + str(g_max))
        print("B: " + str(b_min) + " - " + str(b_max))
        print("=================================")

    return [([g_min, b_min, r_min], [g_max, b_max, r_max])]


if __name__ == '__main__':
    main()
