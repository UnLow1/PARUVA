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
###### VIDEO FROM FILE #########
################################
import numpy as np
import cv2

set_new_colors = False
detect_ball = True
detect_red_players = False
detect_blue_players = False
# template: boundaries = [([G_min, B_min, R_min], [G_max, B_max, R_max])]
# boundaries = [([0, 50, 150], [50, 180, 200])] #[([0, 0, 190], [62, 174, 250])]
boundaries_ball = [([0, 50, 150], [50, 180, 200])]
boundaries_red_players = [([0, 0, 140], [80, 80, 255])]
boundaries_blue_players = [([100, 0, 0], [255, 100, 50])]
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
output_filename = "output"
filename = 'pilkarzyki30.mp4'
frames_without_ball = 35


def main():
    BUFFER_SIZE = 120  # 50 frames for goal detection, 70 frames for replay
    buffer = [[]] * BUFFER_SIZE
    index = 0

    boundaries = set_proper_boundaries()
    cap = cv2.VideoCapture(filename)

    file_counter = 0
    counter = 0

    font = cv2.FONT_ITALIC

    while cap.isOpened():
        ret, frame = cap.read()

        set_score_on_frame(frame, font)

        buffer[index % BUFFER_SIZE] = frame
        index += 1

        # create NumPy arrays from the boundaries
        lower = np.array(boundaries[0][0], dtype="uint8")
        upper = np.array(boundaries[0][1], dtype="uint8")

        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(frame, lower, upper)
        output = cv2.bitwise_and(frame, frame, mask=mask)

        if is_ball_detected(output):
            counter = 0
        else:
            counter += 1
            if counter == frames_without_ball:
                save_buffer_to_file(buffer, index, output_filename, file_counter)
                cv2.putText(frame, 'Replay saved', (700, 550), font, 3, (0, 255, 0), 10, cv2.LINE_AA)
                file_counter += 1
            print(counter)

        cv2.imshow("Pilkarzyki game", frame)

        if set_new_colors:
            cv2.waitKey(0)
            boundaries = set_colors(boundaries)
        else:
            cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()


def is_ball_detected(output):
    output2 = cv2.resize(output, (150, 100))
    for matrix in output2:
        for array in matrix:
            if np.any(array):
                return True
    return False


def save_buffer_to_file(buffer, index, filename, counter):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    output_filename = str(filename) + str(counter) + ".avi"
    out = cv2.VideoWriter(output_filename, fourcc, 20.0, (VIDEO_WIDTH, VIDEO_HEIGHT))
    if index < len(buffer):
        index = 0
    else:
        index += 1
    for i in range(len(buffer)):
        if i:
            out.write(buffer[index % len(buffer)])
        index += 1
    frame = cv2.imread("ericsson_logo.jpg")
    resized_frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
    for i in range(20):
        out.write(resized_frame)
    out.release()


def set_proper_boundaries():
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

    return [([g_min, b_min, r_min], [g_max, b_max, r_max])]


def set_score_on_frame(frame, font):
    cv2.putText(frame, 'RED TEAM       0', (50, 100), font, 3, (0, 0, 255), 10, cv2.LINE_AA)
    cv2.putText(frame, ':', (970, 100), font, 4, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, '1     BLUE TEAM', (1050, 100), font, 3, (255, 0, 0), 10, cv2.LINE_AA)
    # image = cv2.imread("logo.png", cv2.IMREAD_UNCHANGED)
    # transparent_overlay(frame, image, (1750, 920), 0.3)


def transparent_overlay(src, overlay, pos=(0, 0), scale=1):
    """
    :param src: Input Color Background Image
    :param overlay: transparent Image (BGRA)
    :param pos:  position where the image to be blit.
    :param scale : scale factor of transparent image.
    :return: Resultant Image
    """
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape  # Size of foreground
    rows, cols, _ = src.shape  # Size of background Image
    y, x = pos[0], pos[1]  # Position of foreground/overlay image

    # loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            alpha = float(overlay[i][j][3] / 255.0)  # read the alpha channel
            src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
    return src


if __name__ == '__main__':
    main()
