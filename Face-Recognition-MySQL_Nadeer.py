from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import face_recognition
import cv2
import numpy as np
import mysql.connector
from io import BytesIO
from PIL import Image

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setMinimumSize(QtCore.QSize(1200, 633)) 
        MainWindow.setMaximumSize(QtCore.QSize(1200, 633))  
        MainWindow.setBaseSize(QtCore.QSize(1200, 633))

        # Central widget setup
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Add tab widget
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 1200, 600))  

        self.tab1 = QtWidgets.QWidget()
        self.tab1.setObjectName("tab1")

        self.base = QtWidgets.QLabel(self.tab1)
        self.base.setGeometry(QtCore.QRect(175, 40, 400, 351))
        self.base.setText("")
        self.base_image_path = "1.jpg"  # Set default base image path
        self.base.setPixmap(QtGui.QPixmap(self.base_image_path))  # Set default base image
        self.base.setScaledContents(True)
        self.base.setObjectName("base")

        self.chosen = QtWidgets.QLabel(self.tab1)
        self.chosen.setGeometry(QtCore.QRect(625, 40, 400, 351))
        self.chosen.setScaledContents(True)
        self.chosen.setObjectName("chosen")

        self.laad = QtWidgets.QPushButton(self.tab1)
        self.laad.setGeometry(QtCore.QRect(175, 410, 850, 41))
        self.laad.setObjectName("laad")

        
        self.check = QtWidgets.QPushButton(self.tab1)
        self.check.setGeometry(QtCore.QRect(175, 450, 850, 41))
        self.check.setObjectName("check")

        self.message = QtWidgets.QLabel(self.tab1)
        self.message.setGeometry(QtCore.QRect(175, 510, 850, 61))
        font = QtGui.QFont()
        font.setPointSize(36)
        self.message.setFont(font)
        self.message.setAlignment(QtCore.Qt.AlignCenter)
        self.message.setWordWrap(False)
        self.message.setObjectName("message")

        # Add the first tab to the tab widget
        self.tabWidget.addTab(self.tab1, "Face Comparison")

        # Second Tab: Face recognition from database
        self.tab2 = QtWidgets.QWidget()
        self.tab2.setObjectName("tab2")

        
        self.image_viewer = QtWidgets.QLabel(self.tab2)
        self.image_viewer.setGeometry(QtCore.QRect(50, 50, 1100, 351))  # Cover almost full width
        self.image_viewer.setScaledContents(True)
        self.image_viewer.setObjectName("image_viewer")

        # Initialize load and match buttons for second tab
        self.load_button = QtWidgets.QPushButton(self.tab2)
        self.load_button.setGeometry(QtCore.QRect(300, 450, 221, 41))
        self.load_button.setObjectName("load_button")

        self.match_button = QtWidgets.QPushButton(self.tab2)
        self.match_button.setGeometry(QtCore.QRect(600, 450, 221, 41))
        self.match_button.setObjectName("match_button")

        # Add the second tab to the tab widget
        self.tabWidget.addTab(self.tab2, "Face Recognition")

        # Central widget setup
        MainWindow.setCentralWidget(self.centralwidget)

        # Menubar and statusbar setup
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 801, 26))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)

        # Call retranslateUi to set up text for all UI components
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # Button Functionality (connect button clicks to their respective methods)
        self.laad.clicked.connect(self.load_chosen_image)
        self.check.clicked.connect(self.check_images)
        self.load_button.clicked.connect(self.load_image_for_recognition)
        self.match_button.clicked.connect(self.match_face)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Face Recognition and Comparison"))

        # Set button texts
        self.laad.setText(_translate("MainWindow", "Load an image"))
        self.check.setText(_translate("MainWindow", "Check"))
        self.message.setText(_translate("MainWindow", "Result"))
        self.load_button.setText(_translate("MainWindow", "Load Image"))
        self.match_button.setText(_translate("MainWindow", "Match"))

    def load_chosen_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(None, "Choose Image", "", "Image Files (*.jpg *.jpeg *.png)", options=options)
        if file_path:
            self.chosen_image_path = file_path
            pixmap = QtGui.QPixmap(file_path)
            self.chosen.setPixmap(pixmap)

    def check_images(self):
        if not hasattr(self, 'chosen_image_path'):
            self.message.setText("Please load a chosen image.")
            return

        if self.verify_faces(self.base_image_path, self.chosen_image_path):
            self.message.setText("The faces match!")
        else:
            self.message.setText("The faces do not match.")

    def verify_faces(self, image1_path, image2_path):
        encoding_1 = self.encode_face(image1_path)
        encoding_2 = self.encode_face(image2_path)
        if encoding_1 is None or encoding_2 is None:
            QMessageBox.warning(None, "Error", "One of the images doesn't contain a detectable face.")
            return False

        results = face_recognition.compare_faces([encoding_1], encoding_2)
        return results[0]

    def encode_face(self, image_path):
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        if len(face_encodings) > 0:
            return face_encodings[0]
        return None

    def load_image_for_recognition(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(None, "Choose Image", "", "Image Files (*.jpg *.jpeg *.png)", options=options)
        if file_path:
            self.recognition_image_path = file_path
            pixmap = QtGui.QPixmap(file_path)
            self.image_viewer.setPixmap(pixmap)

    def match_face(self):
        if not hasattr(self, 'recognition_image_path'):
            QMessageBox.warning(None, "Error", "Please load an image to match.")
            return
    
        known_face_encodings, known_face_names = self.load_encodings_from_database()
        
        self.recognize_face(self.recognition_image_path, (known_face_encodings, known_face_names))

    def recognize_face(self, image_path, known_encodings_names):
        known_face_encodings, known_face_names = known_encodings_names 
    
        image = face_recognition.load_image_file(image_path)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    
        face_locations = face_recognition.face_locations(image, model="cnn")
        face_encodings = face_recognition.face_encodings(image, face_locations, model="large")
    
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            
            
            padding = 10  
            top = max(0, top - padding)
            right = min(image_bgr.shape[1], right + padding)
            bottom = min(image_bgr.shape[0], bottom + padding)
            left = max(0, left - padding)
            
            
            cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 0, 255), 2)
    
            label_height = 50  
            cv2.rectangle(image_bgr, (left, bottom - label_height), (right, bottom), (0, 0, 255), cv2.FILLED)
    
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(image_bgr, name, (left + 6, bottom - 6), font, 1, (255, 255, 255), 1)
    
        # Convert the BGR image back to RGB for correct display
        rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
        
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    
        
        self.image_viewer.setPixmap(QtGui.QPixmap.fromImage(q_img))

    # Load encodings from the database
    def load_encodings_from_database(self):
        known_face_encodings = []
        known_face_names = []

        # Database connection
        try:
            db_connection = mysql.connector.connect(
                host="localhost",
                user="root",
                password="",  
                database="images_db" 
            )

            cursor = db_connection.cursor()
            cursor.execute("SELECT image_name, image_column FROM images_store")  
            results = cursor.fetchall()

            for name, image_blob in results:
                image = face_recognition.load_image_file(BytesIO(image_blob))  
                face_encodings = face_recognition.face_encodings(image)

                if face_encodings:  
                    known_face_encodings.append(face_encodings[0])  
                    known_face_names.append(name) 

            cursor.close()
            db_connection.close()

            return known_face_encodings, known_face_names

        except mysql.connector.Error as err:
            print(f"Error: {err}")
            return [], [] 

# Main execution
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    
    
    MainWindow.closeEvent = lambda event: app.quit()  
    
    MainWindow.show()
    app.exec_()  


