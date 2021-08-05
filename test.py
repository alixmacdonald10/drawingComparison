import main
import cv2


if __name__ == "__main__":
    
    # file paths
    img1_path = "D:\\Scripts\\drawingComparison\\examples\\17916_A-1.jpg"
    img2_path = "D:\\Scripts\\drawingComparison\\examples\\17916_B-1.jpg"
    # load the two input images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    main.run(img1, img2)