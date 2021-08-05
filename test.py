import main
import cv2
from pdf2image import convert_from_path, convert_from_bytes


if __name__ == "__main__":
    
    # Img path
    img1Path = 'D:\\Scripts\\drawingComparison\\examples\\17916_A'
    img2Path = 'D:\\Scripts\\drawingComparison\\examples\\17916_B'
    imgPaths = [img1Path, img2Path]
    # convert pdf to jpg
    for i in range(0, 2):
        pages = convert_from_path(f'{imgPaths[i]}.pdf')
        for page in pages:
            page.save(f'{imgPaths[i]}.jpg', 'JPEG')

    # load the two input images
    img1 = cv2.imread(f'{img1Path}.jpg')
    img2 = cv2.imread(f'{img2Path}.jpg')
    
    main.run(img1, img2)