import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque
import operator

class EmotionRecognition:
    def __init__(self, model_path='model_resnet_emotion.h5'):
        self.emotion_queue = deque(maxlen=10)
        self.model = load_model(model_path)
        self.emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    def smooth_emotions(self, prediction):
        emotion_values = {emotion: 0.0 for emotion in self.emotions}

        emotion_probability = np.max(prediction)
        emotion_index = np.argmax(prediction)
        emotion = self.emotions[emotion_index]

        # Append the new emotion
        self.emotion_queue.appendleft((emotion_probability, emotion))

        # Average across the queue
        for prob, emo in self.emotion_queue:
            emotion_values[emo] += prob

        average_emotion = max(emotion_values.items(), key=operator.itemgetter(1))[0]
        return average_emotion

    def process_image(self, roi_gray, img):
        image_scaled = cv2.resize(roi_gray, (48, 48))
        image_processed = image_scaled.astype('float32') / 255.0
        image_processed = np.expand_dims(image_processed, axis=(0, -1))

        prediction = self.model.predict(image_processed)
        emotion = self.smooth_emotions(prediction[0])

        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f"Emotion: {emotion}", (50, 450), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('img', img)

    def run(self):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        cap = cv2.VideoCapture(0)

        while True:
            ret, img = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                self.process_image(roi_gray, img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    emotion_recognition = EmotionRecognition()
    emotion_recognition.run()
