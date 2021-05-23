import cv2

faceDetect = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


class Video(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()
        faces = faceDetect.detectMultiScale(frame, 1.3, 5)
        for x, y, w, h in faces:
            x1, y1 = x+w, y+h
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 195, 0), 1)
            cv2.line(frame, (x, y), (x+40, y), (255, 195, 0), 6)
            cv2.line(frame, (x, y), (x, y+40), (255, 195, 0), 6)

            cv2.line(frame, (x1, y), (x1-40, y), (255, 195, 0), 6)
            cv2.line(frame, (x1, y), (x1, y+40), (255, 195, 0), 6)

            cv2.line(frame, (x, y1), (x+40, y1), (255, 195, 0), 6)
            cv2.line(frame, (x, y1), (x, y1-40), (255, 195, 0), 6)

            cv2.line(frame, (x1, y1), (x1-40, y1), (255, 195, 0), 6)
            cv2.line(frame, (x1, y1), (x1, y1-40), (255, 195, 0), 6)
        ret, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()
