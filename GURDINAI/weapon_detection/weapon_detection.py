import cv2
import numpy as np
import pygame
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import imghdr
from twilio.rest import Client

# Load Yolo
net = cv2.dnn.readNet("yolov3_training_2000.weights", "yolov3_testing.cfg")
classes = ["Weapon"]
output_layer_names = net.getUnconnectedOutLayersNames()
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Initialize pygame for sound
pygame.init()
alert_sound_path = "Police Siren Ringtone Download - MobCup.Com.Co.mp3"  # Replace with your MP3 file path
alert_sound = pygame.mixer.Sound(alert_sound_path)

# Email configuration (replace with your actual email credentials and settings)
email_address = "shambhurajerdeshmukh@coep.sveri.ac.in"
email_password = "Shambhu@8625"
smtp_server = "smtp.gmail.com"
smtp_port = 587
owner_email = "shambhurajed9@gmail.com"

# Twilio configuration (replace with your actual Twilio credentials and numbers)
twilio_account_sid = "AC004583e4c1f2fe511a92f9ada4d530d7"
twilio_auth_token = "bbf282ea9ac36afa1a492f491de54244"
twilio_whatsapp_number = "whatsapp:+14155238886"
your_whatsapp_number = "whatsapp:+918625897596"  # Replace with the owner's WhatsApp number

client = Client(twilio_account_sid, twilio_auth_token)

# Loading video or webcam
def get_video_source():
    val = input("Enter file name or press enter to start webcam: \n")
    if val == "":
        val = 0
    return val

cap = cv2.VideoCapture(get_video_source())

# Set frame dimensions to full frame
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Adjust the width as needed
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Adjust the height as needed

# Flag to indicate whether the sound is currently playing
sound_playing = False

while True:
    _, img = cap.read()
    if not _:
        print("Error: Failed to read a frame from the video source.")
        break

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layer_names)

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []
    weapon_detected = False

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "Weapon":
                # Object detected
                center_x = int(detection[0] * img.shape[1])
                center_y = int(detection[1] * img.shape[0])
                w = int(detection[2] * img.shape[1])
                h = int(detection[3] * img.shape[0])

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                weapon_detected = True

    if weapon_detected and not sound_playing:
        print("Weapon detected in frame")
        alert_sound.play()

        # Send WhatsApp alert
        whatsapp_message = "Weapon detected! Check your email for details."
        client.messages.create(
            to=your_whatsapp_number,
            from_=twilio_whatsapp_number,
            body=whatsapp_message
        )

        # Take a picture
        picture_filename = "weapon_detection_picture.jpg"
        cv2.imwrite(picture_filename, img)

        # Send email with the picture attached
        msg = MIMEMultipart()
        msg['From'] = email_address
        msg['To'] = owner_email
        msg['Subject'] = "Weapon Detected"

        body = "A weapon has been detected. Please find the attached picture."
        msg.attach(MIMEText(body, 'plain'))

        with open(picture_filename, 'rb') as attachment:
            image_data = attachment.read()
            image_type = imghdr.what(picture_filename)
            msg.attach(MIMEImage(image_data, image_type))

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(email_address, email_password)
            server.sendmail(email_address, owner_email, msg.as_string())

        sound_playing = True
    elif not weapon_detected:
        sound_playing = False

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
