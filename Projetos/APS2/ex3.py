import cv2
import numpy as np
import matplotlib.pyplot as plt

base = cv2.imread(r"C:\Users\lucca\OneDrive - Insper\Documentos\Insper\6\VisMaq\MachineVision\Projetos\APS2\Figuras_APS2\Fig3_Base.bmp", cv2.IMREAD_GRAYSCALE)
arm1 = cv2.imread(r"C:\Users\lucca\OneDrive - Insper\Documentos\Insper\6\VisMaq\MachineVision\Projetos\APS2\Figuras_APS2\Fig3_Arm1.bmp", cv2.IMREAD_GRAYSCALE)
arm2 = cv2.imread(r"C:\Users\lucca\OneDrive - Insper\Documentos\Insper\6\VisMaq\MachineVision\Projetos\APS2\Figuras_APS2\Fig3_Arm2.bmp", cv2.IMREAD_GRAYSCALE)
data = np.genfromtxt(r"C:\Users\lucca\OneDrive - Insper\Documentos\Insper\6\VisMaq\MachineVision\Projetos\APS2\Figuras_APS2\Robo_Cinematica.csv", delimiter=';')

def create_robot_arm_visualization(base, arm1, arm2, theta1, theta2):
    """
    Creates a visualization of a robot arm by combining base, arm1, and arm2 images
    with specified rotation angles.
    
    Parameters:
    -----------
    base : numpy.ndarray
        The base image of the robot
    arm1 : numpy.ndarray
        The first arm segment image
    arm2 : numpy.ndarray
        The second arm segment image
    theta1 : float
        Rotation angle for arm2 in degrees (default: 0)
    theta2 : float
        Rotation angle for the combined arms in degrees (default: 0)
        
    Returns:
    --------
    numpy.ndarray
        The combined image with all robot parts
    """
    # Binarize the images using thresholding
    _, base_binary = cv2.threshold(base, 127, 255, cv2.THRESH_BINARY)
    _, arm1_binary = cv2.threshold(arm1, 127, 255, cv2.THRESH_BINARY)
    _, arm2_binary = cv2.threshold(arm2, 127, 255, cv2.THRESH_BINARY)

    # Replace original grayscale images with binary versions
    base = base_binary.copy()
    arm1 = arm1_binary
    arm2 = arm2_binary

    # Create a blank canvas for the arms
    vazia = np.ones((1000, 1000), dtype=np.uint8)

    # Place arm1 on the canvas
    arm1_height, arm1_width = arm1.shape
    start_y = 0
    start_x = 0

    for i in range(arm1_height):
        for j in range(arm1_width):
            if arm1[i, j] == 255: 
                vazia[start_y + i, start_x + j] = arm1[i, j]

    # Convert theta1 from degrees to radians
    theta = theta1 * np.pi / 180

    # Create transformation matrix for arm2
    TH01 = np.array([[np.cos(theta), -np.sin(theta), 285],
                    [np.sin(theta), np.cos(theta), 75],
                    [0, 0, 1]], dtype=np.float32)

    # Apply transformation to arm2 and place it on the canvas
    (h, w) = arm2.shape
    for x in range(w):
        for y in range(h):
            if arm2[y, x] == 255:
                p0 = np.array([[x], [y], [1]])
                p1 = np.matmul(TH01, p0)
                u = int(p1[0,0])
                v = int(p1[1,0])

                if u >= 0 and u < vazia.shape[1] and v >= 0 and v < vazia.shape[0]:
                    vazia[v, u] = arm2[y, x]

    arms = vazia

    # Convert theta2 from degrees to radians
    theta2_rad = theta2 * np.pi / 180

    # Create transformation matrix for the combined arms
    TH02 = np.array([[np.cos(theta2_rad), -np.sin(theta2_rad), 140],
                    [np.sin(theta2_rad), np.cos(theta2_rad), 180],
                    [0, 0, 1]], dtype=np.float32)

    # Create a copy of the base image for the result
    result = base.copy()
    
    # Apply transformation to the combined arms and place them on the base
    (h, w) = arms.shape
    for x in range(w):
        for y in range(h):
            if arms[y, x] == 255:
                p0 = np.array([[x], [y], [1]])
                p1 = np.matmul(TH02, p0)
                u = int(p1[0,0])
                v = int(p1[1,0])

                if u >= 0 and u < result.shape[1] and v >= 0 and v < result.shape[0]:
                    result[v, u] = arms[y, x]

    return result



# Loop through each position in the data
for theta1, theta2 in data:
    # Create the robot arm visualization for the current angles
    frame = create_robot_arm_visualization(base, arm1, arm2, theta1, theta2)
    
    # Display the frame
    cv2.imshow('Robot Arm Animation', frame)
    # Wait for a short time between frames
    key = cv2.waitKey(200) & 0xFF
    if key == ord('q'):
        break





