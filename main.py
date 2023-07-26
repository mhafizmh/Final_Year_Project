import sys
import numpy as np
import cv2
import dlib
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog, QRadioButton
from PyQt5.QtCore import QTimer
from keras.models import load_model
from collections import deque


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Create a QLabel widget to display the video feed
        self.label = QLabel(self)
        self.label.setFixedSize(640, 480)
        
        # Create radio buttons for selecting video file or webcam feed
        self.radio_select = QRadioButton("Select Video File")
        self.radio_webcam = QRadioButton("Webcam")
        self.radio_select.setChecked(True)  # Select the "Select Video File" radio button by default
        
        # Create a push button for selecting a video file
        self.button_select = QPushButton("Select Video File")
        self.button_select.clicked.connect(self.select_file)
        
        # Create a push button to start playing the video or webcam feed
        self.button_start = QPushButton("Start")
        self.button_start.clicked.connect(self.start)
    
        # Create a push button to stop playing the video or webcam feed
        self.button_stop = QPushButton("Stop")
        self.button_stop.clicked.connect(self.stop)
        
        
        # Check if the radio button selection has changed
        self.radio_select.clicked.connect(self.radio_button_changed)
        self.radio_webcam.clicked.connect(self.radio_button_changed)
        
        # Create Haar Cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Create a QVBoxLayout widget to hold the QLabel and radio buttons
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.radio_select)
        layout.addWidget(self.radio_webcam)
        layout.addWidget(self.button_select)
        layout.addWidget(self.button_start)
        layout.addWidget(self.button_stop)
        
        # Create a QWidget object and set the QVBoxLayout as its layout
        widget = QWidget()
        widget.setLayout(layout)
        
        # Set the QWidget as the central widget of the QMainWindow
        self.setCentralWidget(widget)
        
        # Create a QTimer object to update the QLabel periodically
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_label)
        
        # Create a VideoCapture object to capture frames from the webcam or video file
        self.cap = None
        
        # Load the model
        self.model = load_model('model.hdf5')
        
        # Create emotion labels
        self.emotion_labels = {0:'Angry', 1:'Happy', 2:'Disgust', 3:'Sad', 4:'Fear'}
        
        # Define the desired input shape
        self.image_rows, self.image_columns, self.image_depth = 64, 64, 96
        
        self.emotion = "Neutral"
        
        # Define a deque to store the last 96 frames
        self.frame_deque = deque(maxlen=self.image_depth)
        
                        
    def select_file(self):
        # Open a file dialog to select a video file
        filename, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi)")
        
        # Create a VideoCapture object to read the selected video file
        if filename:
            self.cap = cv2.VideoCapture(filename)
            self.radio_select.setChecked(True)
        
    def stop(self):
        # Stop the time if it's running
        self.timer.stop()
        
        self.label.setText("Video playback has been stopped")
        
            
    def start(self):
        # Stop the timer if it's running
        self.timer.stop()
        
        # Set the VideoCapture object based on the selected radio button
        if self.radio_select.isChecked():
            if not self.cap:
                # If no video file is selected, show an error message
                self.label.setText("Please select a video file first.")
                return
        else:
            self.cap = cv2.VideoCapture(0)
        
        # Start the timer to update the QLabel
        self.timer.start(30)  # Update the QLabel every 30 milliseconds
        
    def radio_button_changed(self):
        if self.radio_select.isChecked():
            self.reset_label()
            self.cap = None
            self.frame_deque.clear
            self.emotion = "Neutral"
        
        elif self.radio_webcam.isChecked():
            self.reset_label()
            self.cap = None
            self.frame_deque.clear
            self.emotion = "Neutral"
            
    def reset_label(self):
        self.label.setText("No video selected")
        self.label.setPixmap(QPixmap())
        self.timer.stop()
                
        
    def update_label(self):
        self.emotion = 'None'
        # Read a frame from the VideoCapture object
        ret, frame = self.cap.read()
        
        if not ret:
            # If no frame is returned, stop the timer and set the QLabel text to "End of Video"
            self.timer.stop()
            self.label.setText("End of Video")
            return
                        
        # Resize the frame to fit the label's size
        frame = cv2.resize(frame, (self.label.width(), self.label.height()))
    
        # Convert the frame from BGR to RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize the frame to the desired size
        frame2 = cv2.resize(frame, (self.image_rows, self.image_columns))

        # Convert the frame to grayscale
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Add the frame to the deque
        self.frame_deque.append(frame2)
        
        # If we have enough frames, predict the label
        if len(self.frame_deque) == self.image_depth:
            # Convert the deque to a numpy array
            frame_array = np.array(self.frame_deque)
            
            # Add an extra dimension for the channel
            frame_array = frame_array.reshape(1, 1, self.image_rows, self.image_columns, self.image_depth)
            
            # Normalize the input
            frame_array = frame_array.astype('float32')
            frame_array -= np.mean(frame_array)
            frame_array /= np.max(frame_array)
            
            # Predict the label
            predict_output = self.model.predict(frame_array)
            predict_index = np.argmax(predict_output[0])
            
            # Get the label base on the index
            self.emotion = self.emotion_labels[predict_index]
            print(self.emotion)    
            self.frame_deque.popleft()
            print(len(self.frame_deque))
            
        # Convert the frame from BGR to RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Create a QPixmap object from the frame and set it as the QPixmap of the QLabel
        
        pixmap = QPixmap.fromImage(QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888))
        self.label.setPixmap(pixmap)
    
        
        # Detect faces in the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
        # Draw bounding boxes around the faces
        for (x, y, w, h) in faces:
            cv2.putText(frame, self.emotion, (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                
        # Convert the frame to QImage format and display it on the label
        height, width, channel = frame.shape
        bytesPerLine = 3 * width
        qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(qImg))

                
if __name__ == '__main__':
    # Create a QApplication instance
    app = QApplication(sys.argv)

    # Create a MainWindow instance
    window = MainWindow()
    window.show()

    # Run the event loop
    sys.exit(app.exec_())
