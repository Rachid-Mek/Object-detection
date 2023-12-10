import cv2
import numpy as np
from Fonctions import *
from Cam_detection import *


def game():
    """ Launch the game and detect the object in the image captured by the camera and move the car according to the object position
    """
    vitesse = 300 # speed of the car
    cap = cv2.VideoCapture(0) # Launch the camera
    if not cap.isOpened(): # Check if the camera is opened
        print("Erreur de capture") # Print an error message
        exit(0) # Exit the program

    car = cv2.imread('Images/car.png', cv2.IMREAD_UNCHANGED) # Read the image of the car with the alpha channel
    contour = cv2.imread('Images/contour.jpg', cv2.IMREAD_COLOR) # Read the image of the contour of the game

    alpha_channel = car[:, :, 3] # Extract the alpha channel of the car image 

    contour = cv2.resize(contour, (500, 500))  # Resize the contour image to the size of the game window which is 500x500

    obstacle_pos_x = np.random.randint(0, contour.shape[1] - 30) # Generate a random position for the obstacle
    obstacle_pos_y = 0 # Initial position of the obstacle

    obstacle_pos_x2 = np.random.randint(0, contour.shape[1] - 30) # Generate a random position for the obstacle
    obstacle_pos_y2 = 0 # Initial position of the obstacle

    car_pos_x = (contour.shape[1] - car.shape[1]) // 2  # Initial position of the car in the middle of the game window
    car_pos_y = contour.shape[0] - car.shape[0]  # Initial position of the car at the bottom of the game window

    prev_obj_center_x = 0 # Previous object center x position

    while True: # Loop until the user press the key 'q'
        ret, frame = cap.read() # Capture the frame
        cv2.flip(frame,1,frame) # Flip the frame horizontally
        if not ret: # Check if the frame is captured
            print("Error in image read") # Print an error message
            break

        resized_frame = resize_image_3d(frame, 0.1) # Resize the frame to a lower resolution for faster processing

        # Detect the object and get its information
        obj_info = detect_object(resized_frame) # Detect the object in the image captured by the camera

        # Display the camera window with circles and points
        if obj_info and len(obj_info[2]) > 0: # Check if the object is detected
            _, _, obj_coords = obj_info # Get the object coordinates
            cv2.circle(frame, (int(obj_coords[0][0] * 10), int(obj_coords[0][1] * 10)), 20, (0, 255, 0), 5) # Draw a circle around the object
            cv2.putText(frame, "x: {}, y: {}".format(int(obj_coords[0][0] * 10), int(obj_coords[0][1] * 10)),
                        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 4)

        cv2.imshow('Camera', frame) # Show the camera window

        # Continue with the game window
        if obj_info and len(obj_info[2]) > 0: # Check if the object is detected
            blurred_img, _, obj_coords = obj_info # Get the object coordinates

            fenetre = contour.copy() # Copy the contour image

            # Dessiner la voiture sur la fenêtre avec la transparence correcte
            for c in in_range(3):
                car_pos_x = max(0, min(car_pos_x, fenetre.shape[1] - car.shape[1])) # Check if the car exceeds the left boundary

                left_boundary = max(car_pos_x, 0)  # Check if the car exceeds the left boundary
                right_boundary = min(car_pos_x + car.shape[1], fenetre.shape[1]) # Check if the car exceeds the right boundary

                fenetre[car_pos_y:car_pos_y + car.shape[0], left_boundary:right_boundary, c] = (
                    car[:, car_pos_x - left_boundary:car_pos_x - left_boundary + (right_boundary - left_boundary), c] * 
                    (alpha_channel / 255.0) +
                    fenetre[car_pos_y:car_pos_y + car.shape[0], left_boundary:right_boundary, c] *
                    (1.0 - alpha_channel / 255.0)
                ).astype(np.uint8) # Draw the car on the game window with the correct transparency
            
            # Draw the obstacles on the game window (red rectangles)
            fenetre[obstacle_pos_y:obstacle_pos_y + 40, obstacle_pos_x:obstacle_pos_x + 30] = [0, 0, 255]  # rouge
            fenetre[obstacle_pos_y2:obstacle_pos_y2 + 40, obstacle_pos_x2:obstacle_pos_x2 + 30] = [0, 0, 255]  # rouge

            # Update the position of the obstacles
            obstacle_pos_y += 5   # vitesse de descente ici
            obstacle_pos_y2 += 5   # vitesse de descente ici

            # Réinitialiser la position de l'obstacle lorsqu'il atteint le bas de l'image
            if obstacle_pos_y > contour.shape[0]:
                obstacle_pos_y = 0
                obstacle_pos_x = np.random.randint(0, contour.shape[1] - 30)

            if obstacle_pos_y2 > contour.shape[0]:
                obstacle_pos_y2 = 0
                obstacle_pos_x2 = np.random.randint(0, contour.shape[1] - 30)

            # Déplacer la voiture en fonction des coordonnées de l'objet
            obj_center_x = obj_coords[0][0]
            print(obj_center_x)

            # Vérifier la collision avec les obstacles
            collision1 = check_collision(car_pos_x, car_pos_y, car.shape[1], car.shape[0],
                                         obstacle_pos_x, obstacle_pos_y, 30, 40) 

            collision2 = check_collision(car_pos_x, car_pos_y, car.shape[1], car.shape[0],
                                         obstacle_pos_x2, obstacle_pos_y2, 30, 40)

            # Si collision, afficher "Game Over" et fermer la fenêtre
            if collision1 or collision2:
                cv2.putText(fenetre, "Game Over", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
                cv2.imshow('Racing Game', fenetre)
                cv2.waitKey(2500)  # Attendre 2.5 secondes
                break

            # Detect if there are any objects
            if len(obj_coords) > 0:
                point = tuple(map(int, obj_coords[0]))  # Convert to tuple of integers
                cv2.circle(frame, (point[0] * 10, point[1] * 10), 150, (0, 255, 0), 5)
                cv2.putText(frame, "x: {}, y: {}".format(point[0] * 10, point[1] * 10),
                            (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 4)
          
            if 0 <= obj_center_x * 10 <= 170:
                # Move to the left
                car_pos_x -= 10
            elif obj_center_x * 10 > 420:
                # Move to the right
                car_pos_x += 10
            # Afficher la fenêtre de jeu
            cv2.imshow('Racing Game', fenetre)

        # Capturer la touche pressée (pour quitter la boucle si nécessaire)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Libérer les ressources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    game()