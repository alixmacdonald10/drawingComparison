import cv2
from skimage.metrics import structural_similarity as compare_ssim
import argparse
import imutils
from pdf2image import convert_from_path



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

    img1Path = convert_from_path(args["first"])
    img2Path = convert_from_path(args["second"])
    imgPaths = [img1Path, img2Path]
    # convert pdf to jpg
    for i in range(0, 1):
        pages = convert_from_path(f'{imgPaths[i]}.pdf')
        for page in pages:
            page.save(f'{imgPaths[i]}.jpg', 'JPEG')
            
    # load the two input images
    img1 = cv2.imread(f'{img1Path}.jpg')
    img2 = cv2.imread(f'{img2Path}.jpg')
    
    return image1, image2


def SSIM(image1, image2):
    # compute the Structural Similarity Index (SSIM)
    # The score represents the structural similarity index between the two input images. 
    # This value can fall into the range [-1, 1] with a value of one being a “perfect match”.
    # convert the images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
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
    cv2.imwrite("D:\\Scripts\\drawingComparison\\results\\Revised_Bounding_Box.jpg", img2)
    cv2.imwrite("D:\\Scripts\\drawingComparison\\results\\Difference.jpg", diff)
    

if __name__ == "__main__":

    image1, image2 = loadImage()
    
    run(image1, image2)