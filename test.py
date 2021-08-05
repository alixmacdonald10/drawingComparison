import main
import cv2


if __name__ == "__main__":
    
    # file paths
    img1_path = "D:\\Scripts\\drawingComparison\\examples\\17916_A-1.jpg"
    img2_path = "D:\\Scripts\\drawingComparison\\examples\\17916_B-1.jpg"
    # load the two input images
    image1 = cv2.imread(img1_path)
    image2 = cv2.imread(img2_path)
    # convert the images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    
    main.run(gray1, gray2)