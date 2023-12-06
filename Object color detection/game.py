import cv2
import numpy as np
from Fonctions import *
from Cam_detection import *

# Charger l'image de la voiture avec canal alpha
car = cv2.imread('Object color detection/Images/car.png', cv2.IMREAD_UNCHANGED)
# Charger l'image de contour
contour = cv2.imread('Object color detection/Images/contour.jpg', cv2.IMREAD_COLOR)

# Extraire le canal alpha de l'image de la voiture (le contoure noir)
alpha_channel = car[:, :, 3] 

# Resize de l'image de contour
contour = cv2.resize(contour, (500, 500))

# Position initiale de l'obstacle
obstacle_pos_x = np.random.randint(0, contour.shape[1] - 30) 
obstacle_pos_y = 0

obstacle_pos_x2 = np.random.randint(0, contour.shape[1] - 30)
obstacle_pos_y2 = 0

# Position initiale de la voiture
car_pos_x = (contour.shape[1] - car.shape[1]) // 2
car_pos_y = contour.shape[0] - car.shape[0]

# Boucle principale du jeu
while True:
    # Copier l'image de contour
    fenetre = contour.copy()

    # Dessiner la voiture sur la fenêtre avec la transparence correcte
    for c in in_range(0, 3):
        fenetre[car_pos_y:car_pos_y + car.shape[0], car_pos_x:car_pos_x + car.shape[1], c] = (car[:, :, c] * (alpha_channel / 255.0) +
             fenetre[car_pos_y:car_pos_y + car.shape[0], car_pos_x:car_pos_x + car.shape[1], c] *
             (1.0 - alpha_channel / 255.0)).astype(np.uint8)

    # Dessiner l'obstacle (rectangle rouge)
    fenetre[obstacle_pos_y:obstacle_pos_y + 40, obstacle_pos_x:obstacle_pos_x + 30] = [0, 0, 255]  # rouge
    fenetre[obstacle_pos_y2:obstacle_pos_y2 + 40, obstacle_pos_x2:obstacle_pos_x2 + 30] = [0, 0, 255]  # rouge

    # Mettre à jour la position de l'obstacle
    obstacle_pos_y += 5  # vitesse de descente ici
    obstacle_pos_y2 += 5  # vitesse de descente ici

    # Réinitialiser la position de l'obstacle lorsqu'il atteint le bas de l'image
    if obstacle_pos_y > contour.shape[0]:
        obstacle_pos_y = 0
        obstacle_pos_x = np.random.randint(0, contour.shape[1] - 30)

    if obstacle_pos_y2 > contour.shape[0]:
        obstacle_pos_y2 = 0
        obstacle_pos_x2 = np.random.randint(0, contour.shape[1] - 30)

    cv2.imshow('Racing Game', fenetre)

    # Attendre une petite quantité de temps pour simuler le mouvement (peut être ajusté)
    cv2.waitKey(10)

    # Capturer la touche pressée (pour quitter la boucle si nécessaire)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('a') or key == ord('A'):  # 'a' ou 'A' pour déplacer la voiture à gauche
        car_pos_x = move_left(car_pos_x)
    elif key == ord('d') or key == ord('D'):  # 'd' ou 'D' pour déplacer la voiture à droite
        car_pos_x = move_right(car_pos_x, fenetre.shape[1])
# Libérer les ressources
cv2.destroyAllWindows()