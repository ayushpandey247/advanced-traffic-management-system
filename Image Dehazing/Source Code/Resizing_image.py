import cv2

def Resizing_image(HazeImg):
    # Split the channels of the input image
    Channels = cv2.split(HazeImg)
    
    # Check if Channels is empty
    if not Channels:
        print("Error: Could not split channels of input image")
        return HazeImg
    
    # Resize the image
    rows, cols = Channels[0].shape
    HazeImg = cv2.resize(HazeImg, (int(0.7 * cols), int(0.7 * rows)))

    return HazeImg
