from realsense_camera import RealSenseCamera
import cv2

def main():
    cam = RealSenseCamera(save_frames=True)  # geen disk writes
    frame = cam.capture()
    if frame is not None:
        cv2.imshow("Processed Frame", frame)
        print("Frame shape:", frame.shape)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
