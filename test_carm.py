import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)
ok, frame = cap.read()
while ok:

    cv2.imshow("Video", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    ok, frame = cap.read()
