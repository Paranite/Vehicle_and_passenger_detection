import cv2 as cv


def getCameraNum():
    i = 0
    while True:
        # print(i)
        cap = cv.VideoCapture(i)
        if cap is None or not cap.isOpened():
            print("Detected cameras: {}".format(i))
            break
        i += 1
    return i

    # while True:
    #     temp, frame = cap.read()
    #     # print(temp)
    #     original_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    #     original_frame = cv.cvtColor(original_frame, cv.COLOR_BGR2RGB)
    #     # print(original_frame)
    #     cv.imshow('output', original_frame)
    #     if cv.waitKey(25) & 0xFF == ord("q"):
    #         cv.destroyAllWindows()
    #         break
