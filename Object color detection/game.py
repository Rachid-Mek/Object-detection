import cv2
import numpy as np
from Fonctions import *
from Cam_detection import *

def game():
    # Initialiser le score
    score = 0

    # Capturer la vidéo depuis la caméra
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur de capture")
        exit(0)

    # Charger l'image de la voiture avec canal alpha
    car = cv2.imread('Object color detection/Images/car.png', cv2.IMREAD_UNCHANGED)

    # Extraire le canal alpha de l'image de la voiture (le contour noir)
    alpha_channel = np.zeros(car.shape[:2], dtype=car.dtype)
    for i in range(car.shape[0]):
        for j in range(car.shape[1]):
            alpha_channel[i, j] = car[i, j, 3] 

    # Charger l'image de contour
    contour = cv2.imread('Object color detection/Images/contour.jpg', cv2.IMREAD_COLOR)
    
    # Resize de l'image de contour
    contour = resize_image_3d(contour, 0.4)

    # Position initiale de l'obstacle
    obstacle_pos_x = np.random.randint(0, contour.shape[1] - 30) 
    obstacle_pos_y = 0

    obstacle_pos_x2 = np.random.randint(0, contour.shape[1] - 30) 
    obstacle_pos_y2 = 0

    # Position initiale de la voiture
    car_pos_x = (contour.shape[1] - car.shape[1]) // 2  # la voiture se positionne au milieu de l'image
    car_pos_y = contour.shape[0] - car.shape[0]  # la voiture se positionne en bas de l'image

    # Définir la vitesse initiale de descente
    vitesse_descente = 6

    while True:
        # Capturer le frame de la caméra
        ret, frame = cap.read()
        cv2.flip(frame, 1, frame)
        if not ret:
            print("Erreur de lecture d'image")
            break

        # Redimensionner le frame
        resized_frame = resize_image_3d(frame, 0.1)

        # Détecter l'objet et obtenir ses informations
        obj_info = detect_object(resized_frame)

        # Afficher la fenêtre de la caméra avec des cercles et des points
        if obj_info and len(obj_info[2]) > 0:
            _, _, obj_coords = obj_info
            cv2.circle(frame, (int(obj_coords[0][0] * 10), int(obj_coords[0][1] * 10)), 20, (0, 255, 0), 5)
            cv2.putText(frame, "x: {}, y: {}".format(int(obj_coords[0][0] * 10), int(obj_coords[0][1] * 10)),
                        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 4)

        # Afficher la fenêtre de la caméra
        cv2.imshow('Camera', frame)

        # Continuer avec la fenêtre du jeu
        if obj_info and len(obj_info[2]) > 0:
            blurred_img, _, obj_coords = obj_info
            # reinitialiser la fenêtre du jeu avec le contour de l'image
            fenetre = contour.copy() 
            
            # Ajuster la position de la voiture pour ne pas dépasser les limites
            car_pos_x = max(0, min(car_pos_x, fenetre.shape[1] - car.shape[1]))

            # Vérifier si la voiture dépasse la limite gauche de la fenêtre
            left_boundary = max(car_pos_x, 0)
            
            # Vérifier si la voiture dépasse la limite droite de la fenêtre
            right_boundary = min(car_pos_x + car.shape[1], fenetre.shape[1])
           
            # Opérations pixel par pixel pour mélanger l'image de la voiture sur la fenêtre du jeu avec transparence
            for c in in_range(3):
                for i in in_range(car.shape[0]):
                    for j in in_range(right_boundary - left_boundary):
                        fenetre[car_pos_y + i, left_boundary + j, c] = (
                            car[i, car_pos_x - left_boundary + j, c] * (alpha_channel[i, car_pos_x - left_boundary + j] / 255.0) +
                            fenetre[car_pos_y + i, left_boundary + j, c] * (1.0 - alpha_channel[i, car_pos_x - left_boundary + j] / 255.0)
                        ).astype(np.uint8)
            
            # Définir la couleur rouge
            color_red = [0, 0, 255]

            # Dessiner l'obstacle 1 (rectangle rouge)
            for i in range(40):
                for j in range(30):
                    if 0 <= obstacle_pos_y + i < fenetre.shape[0] and 0 <= obstacle_pos_x + j < fenetre.shape[1]:
                        fenetre[int(obstacle_pos_y + i), int(obstacle_pos_x + j)] = color_red

            # Dessiner l'obstacle 2 (rectangle rouge)
            for i in range(40):
                for j in range(30):
                    if 0 <= obstacle_pos_y2 + i < fenetre.shape[0] and 0 <= obstacle_pos_x2 + j < fenetre.shape[1]:
                        fenetre[int(obstacle_pos_y2 + i), int(obstacle_pos_x2 + j)] = color_red

            # Mettre à jour la position de l'obstacle
            obstacle_pos_y += vitesse_descente  # vitesse de descente ici
            obstacle_pos_y2 += vitesse_descente  # vitesse de descente ici
            vitesse_descente += 0.2  # accélération de la vitesse de descente

            # Réinitialiser la position de l'obstacle lorsqu'il atteint le bas de l'image
            if obstacle_pos_y > contour.shape[0]:
                obstacle_pos_y = 0
                obstacle_pos_x = np.random.randint(0, contour.shape[1] - 30)

            if obstacle_pos_y2 > contour.shape[0]:
                obstacle_pos_y2 = 0
                obstacle_pos_x2 = np.random.randint(0, contour.shape[1] - 30)

            # Mettre à jour le score
            score += 1

            # Afficher le score et la vitesse en haut à droite de la fenêtre
            cv2.putText(fenetre, f"Score: {score}", (fenetre.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(fenetre, f"Vitesse: {vitesse_descente:.2f}", (fenetre.shape[1] - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Déplacer la voiture en fonction des coordonnées de l'objet
            obj_center_x = obj_coords[0][0]

            # Vérifier la collision avec les obstacles
            collision1 = check_collision(car_pos_x, car_pos_y, car.shape[1], car.shape[0],
                                         obstacle_pos_x, obstacle_pos_y, 30, 40) 

            collision2 = check_collision(car_pos_x, car_pos_y, car.shape[1], car.shape[0],
                                         obstacle_pos_x2, obstacle_pos_y2, 30, 40)

            # Si collision, afficher "Game Over" et fermer la fenêtre
            if collision1 or collision2:
                cv2.putText(fenetre, "Game Over", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
                cv2.imshow('brick racing game', fenetre)
                cv2.waitKey(2500)  # Attendre 2.5 secondes
                break

            # Détecter s'il y a des objets
            if len(obj_coords) > 0:
                point = tuple(map(int, obj_coords[0]))  # Convertir en tuple d'entiers
                cv2.circle(frame, (point[0] * 10, point[1] * 10), 150, (0, 255, 0), 5)
                cv2.putText(frame, "x: {}, y: {}".format(point[0] * 10, point[1] * 10),
                            (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 4)
          
            # Déplacer la voiture en fonction de la position de l'objet
            if 0 <= obj_center_x * 10 <= 170:
                # Déplacer vers la gauche
                car_pos_x -= 10
            elif obj_center_x * 10 > 420:
                # Déplacer vers la droite
                car_pos_x += 10

            # Afficher la fenêtre du jeu
            cv2.imshow('brick racing game', fenetre)

        # Capturer la touche pressée (pour quitter la boucle si nécessaire)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Libérer les ressources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    game()