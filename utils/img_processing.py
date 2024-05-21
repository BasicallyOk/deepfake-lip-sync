import cv2
import numpy as np


def mask_image(img_path: str):
    # Read an input image as a gray image
    img = cv2.imread(img_path)
    # create a mask
    mask = np.zeros(img.shape[:2], np.uint8)
    mask = cv2.rectangle(mask, (0, 0), (255, 127), 255, -1)

    # compute the bitwise AND using the mask
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    # display the mask, and the output image
    #cv2.imshow('Masked Image', masked_img)

    # save the masked image.
    # cv2.imwrite("masked", masked_img)
    cv2.waitKey(0)
    return masked_img
    


if __name__ == "__main__":
    masked = mask_image("/home/khoa/Documents/Projects/QMIND/deepfake-lip-sync/test_face.png")