import cv2
import glob
import csv
import os

eye_cascPath = 'Closed-Eye-Detection-with-opencv/haarcascade_eye_tree_eyeglasses.xml'  # eye detect model
face_cascPath = 'Closed-Eye-Detection-with-opencv/haarcascade_frontalface_alt.xml'  # face detect model
faceCascade = cv2.CascadeClassifier(face_cascPath)
eyeCascade = cv2.CascadeClassifier(eye_cascPath)

directory = '/home/rajeshkumar/Documents/Rishitha/open-eye-detector/demo_images/test_data'  # Replace with the path to your directory
csv_filename = 'eye_status.csv'  # CSV file name

# Open CSV file for writing
with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['File Name', 'Eye Status'])  # Write header row

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.png')):
                image_path = os.path.join(root, file)
                img = cv2.imread(image_path)

                frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Detect faces in the image
                faces = faceCascade.detectMultiScale(
                    frame,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    # flags = cv2.CV_HAAR_SCALE_IMAGE
                )
                # print("Found {0} faces!".format(len(faces)))
                if len(faces) > 0:
                    # Draw a rectangle around the faces
                    for (x, y, w, h) in faces:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    frame_tmp = img[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]:1, :]
                    frame = frame[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]:1]
                    eyes = eyeCascade.detectMultiScale(
                        frame,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30),
                        # flags = cv2.CV_HAAR_SCALE_IMAGE
                    )
                    if len(eyes) == 0:
                        eye_status = 'Closed'
                        flag = 2
                    else:
                        eye_status = 'Open'
                        flag = 1
                    frame_tmp = cv2.resize(frame_tmp, (400, 400), interpolation=cv2.INTER_LINEAR)
                    cv2.imshow('Face Recognition', frame_tmp)
                else:
                    eye_status = 'No face detected'
                    flag = 0

                # Write file name and eye status to CSV
                csv_writer.writerow([file, flag])
                print(f'File: {file}, Eye Status: {eye_status}')

                # waitkey = cv2.waitKey(0)  # Change waitkey to wait for user input before moving to the next image
                # if waitkey == ord('q') or waitkey == ord('Q'):
                    # cv2.destroyAllWindows()
                    # break
              