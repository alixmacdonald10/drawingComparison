import cv2
from skimage.metrics import structural_similarity as compare_ssim
import argparse
import imutils
import numpy as np

 

def run(img1, img2):
    diffImg = SSIM(img1, img2)
    findDiff(diffImg, img1, img2)
    
    
def loadImage():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--first", required=True,
        help="first input image")
    ap.add_argument("-s", "--second", required=True,
        help="second")
    args = vars(ap.parse_args())
    
    # load the two input images
    image1 = cv2.imread(args["first"])
    # convert the images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.imread(args["second"])
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    return gray1, gray2


def SSIM(gray1, gray2):
    # compute the Structural Similarity Index (SSIM)
    # The score represents the structural similarity index between the two input images. 
    # This value can fall into the range [-1, 1] with a value of one being a “perfect match”.
    (score, diff) = compare_ssim(gray1, gray2, full=True)
    diff = (diff * 255).astype("uint8")
    print(f"SSIM: {score}")
    
    return diff


def findDiff(diff, img1, img2):
    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # show and save the output images
    # cv2.imshow("Original", img1)
    normed_img1 = np.zeros(np.shape(img1))
    img1 = cv2.normalize(img1, normed_img1, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite("D:\\Scripts\\drawingComparison\\examples\\Original_Bounding_Box.jpg", img1)
    # cv2.imshow("Revised", img2)
    cv2.imwrite("D:\\Scripts\\drawingComparison\\examples\\Revised_Bounding_Box.jpg", img2)
    # cv2.imshow("Difference", diff)
    cv2.imwrite("D:\\Scripts\\drawingComparison\\examples\\Difference.jpg", diff)
    # cv2.imshow("Thresh", thresh)
    cv2.imwrite("D:\\Scripts\\drawingComparison\\examples\\Thresh.jpg", thresh)
    # cv2.waitKey(0)  
    

if __name__ == "__main__":

    image1, image2 = loadImage()
    run(image1, image2)