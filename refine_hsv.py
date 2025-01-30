import cv2
import numpy as np

import cv2
import numpy as np

def create_mask_from_hsv(image):
    """
    Creates a binary mask from an BGR image using the provided HSV range.
    Args:
        image (numpy.ndarray): BGR image of size 256x256.
        lower_hsv (numpy.ndarray): Lower bound of HSV range in the form of [H, S, V].
        upper_hsv (numpy.ndarray): Upper bound of HSV range in the form of [H, S, V].
    Returns:
        mask (numpy.ndarray): Binary mask where pixels within the range are 255 and others are 0.
    """
    # Convert BGR image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a mask within the specified range
    lower_red1 = np.array([0, 120, 70], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([170, 120, 70], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([30, 255, 255], dtype=np.uint8)

    lower_blue = np.array([100, 150, 0], dtype=np.uint8)
    upper_blue = np.array([140, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask3 = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    mask4 = cv2.inRange(hsv_image, lower_blue, upper_blue)

    mask = mask1 | mask2 | mask3 | mask4
    print("mask min:", mask.min(), "max:", mask.max(), "sum:", mask.sum())
    cv2.imwrite("debug_mask.png", mask)

    return mask

def apply_mask(image, mask):
    """
    Applies a mask to the image, showing only the parts of the image where the mask is 255.
    Args:
        image (numpy.ndarray): BGR image of size 256x256.
        mask (numpy.ndarray): Binary mask of size 256x256.
    Returns:
        masked_image (numpy.ndarray): Image with the mask applied.
    """
    # Apply the mask to the image
    mask = (mask>0).astype(np.float32)
    masked_image = image * mask[..., None]

    return masked_image.astype(np.uint8)

def display_image(image, title='Image'):
    """
    Displays an image in a window.
    Args:
        image (numpy.ndarray): Image to be displayed.
        title (str): Title of the window.
    """
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main(image_path):
    # Load the image (BGR format)
    image = cv2.imread(image_path)

    # Resize the image to 256x256
    image = cv2.resize(image, (256, 256))

    # Create the mask
    mask = create_mask_from_hsv(image)

    # Apply the mask to the image
    masked_image = apply_mask(image, mask)

    # Display the original and masked images
    display_image(image, title='Original Image')
    display_image(masked_image, title='Masked Image')

if __name__ == '__main__':
    # Path to the BGR image
    image_path = './pics/color_pic_01-23_23-36.png'
    main(image_path)
