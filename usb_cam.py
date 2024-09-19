import cv2

# import stereoigev
#
# disparity_calculator = stereoigev.DisparityCalculator(args=args)

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    H, W = frame.shape[:2]
    half_W = W // 2
    left = frame[:, :half_W, :]
    right = frame[:, half_W:, :]

    cv2.imshow("left", left)
    # disparity = disparity_calculator.calc_by_bgr(left.copy(), right.copy())
    # disp = np.round(disparity * 256).astype(np.uint16)
    # colored = cv2.applyColorMap(cv2.convertScaleAbs(disp, alpha=0.01), cv2.COLORMAP_JET)
    # cv2.imshow("IGEV", colored)
    key = cv2.waitKey(100)
