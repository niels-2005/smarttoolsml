import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


def predictions_with_webcam(model: tf.keras.Model, classes: list, img_size: tuple[int, int] = (64, 64)):
    """_summary_

    Args:
        model (tf.keras.Model): _description_
        classes (list): _description_
        img_size (tuple[int, int], optional): _description_. Defaults to (64, 64).

    Example usage:
        model = load_model("ASL)
        classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        img_size = (64, 64)
        predictions_with_webcam(model=model, classes=classes, img_size=img_size)
    """
    model = load_model('ASL')

    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Kann die Kamera nicht öffnen")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Bild vorverarbeiten wie während des Trainings
        resized_frame = cv2.resize(frame, img_size)  # Angenommen, Ihr Modell erwartet 64x64 Bilder
        normalized_frame = resized_frame / 255.0
        reshaped_frame = np.expand_dims(normalized_frame, axis=0)

        # Vorhersage treffen
        prediction = model.predict(reshaped_frame)
        class_index = np.argmax(prediction)
        pred_class = classes[class_index]
        confidence = np.max(prediction)

        # Zeigen Sie die Vorhersage im Bild an
        cv2.putText(frame, f"Klasse: {pred_class}, Konfidenz: {confidence:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Webcam Live', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()