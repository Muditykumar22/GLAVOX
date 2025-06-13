import cv2
import numpy as np
from video_processing.hand_detection import HandDetector
from video_processing.gesture_recognition import GestureRecognizer
from text_to_speech.speech_synthesis import SpeechSynthesizer
import sounddevice as sd

def convert_gesture_to_text(gesture):
    # Map gesture indices to text
    gesture_map = {
        0: "A", 1: "B", 2: "C", 3: "D", 4: "E",
        5: "F", 6: "G", 7: "H", 8: "I", 9: "J",
        10: "K", 11: "L", 12: "M", 13: "N", 14: "O",
        15: "P", 16: "Q", 17: "R", 18: "S", 19: "T",
        20: "U", 21: "V", 22: "W", 23: "X", 24: "Y",
        25: "Z"
    }
    return gesture_map.get(gesture, "Unknown")

def play_audio(audio):
    # Play audio using sounddevice
    sd.play(audio, samplerate=16000)
    sd.wait()

def main():
    # Initialize components
    hand_detector = HandDetector()
    gesture_recognizer = GestureRecognizer('sign_language_model.pth', num_classes=26)
    speech_synthesizer = SpeechSynthesizer('tts_model.pth', input_dim=256, hidden_dim=512, output_dim=80)

    # Initialize video capture
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect hands
        frame, hand_landmarks = hand_detector.detect_hands(frame)

        if hand_landmarks:
            # Recognize gesture
            gesture = gesture_recognizer.predict(frame)
            
            # Convert gesture to text
            text = convert_gesture_to_text(gesture)
            
            # Display text on frame
            cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Synthesize speech
            audio = speech_synthesizer.synthesize(text)
            play_audio(audio)

        # Display frame
        cv2.imshow('Sign Language Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 