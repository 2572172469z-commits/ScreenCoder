from os.path import join as pjoin
import cv2
import os
import multiprocessing


def resize_height_by_longest_edge(img_path, resize_length=800):
    org = cv2.imread(img_path)
    height, width = org.shape[:2]
    if height > width:
        return resize_length
    else:
        return int(resize_length * (height / width))


def nothing(x):
    pass


if __name__ == '__main__':
    # Set multiprocessing start method for macOS compatibility
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # It's OK if it's already set

    '''
    This script is for testing component detection (ip) with adjustable parameters.
    Press 'q' in an image window to quit.
    
    - min-grad: gradient threshold to produce binary map
    - min-ele-area: minimum area for selected elements
    '''
    key_params = {'min-grad': 4, 'ffl-block': 5, 'min-ele-area': 25,
                  'merge-contained-ele': True, 'remove-bar': True}

    # set input image path
    input_path_img = 'data/test1.png'
    output_root = 'data'

    resized_height = resize_height_by_longest_edge(input_path_img)

    # Classification is disabled by default in this testing script.
    # Set to True if you have a trained model and want to test classification.
    is_clf = False

    # Create window and trackbars for real-time parameter adjustment
    cv2.namedWindow('parameters')
    cv2.createTrackbar('min-grad', 'parameters', key_params['min-grad'], 20, nothing)
    cv2.createTrackbar('min-ele-area', 'parameters', key_params['min-ele-area'], 200, nothing)

    # Main loop for component detection
    while True:
        # Read current trackbar positions
        key_params['min-grad'] = cv2.getTrackbarPos('min-grad', 'parameters')
        key_params['min-ele-area'] = cv2.getTrackbarPos('min-ele-area', 'parameters')

        # Import the component detection module
        import detect_compo.ip_region_proposal as ip
        os.makedirs(pjoin(output_root, 'ip'), exist_ok=True)

        # Set up classifier if enabled
        classifier = None
        if is_clf:
            classifier = {'Elements': CNN('Elements')}
            from cnn.CNN import CNN

        # Run component detection with current parameters
        ip.compo_detection(input_path_img, output_root, key_params,
                           classifier=classifier, resize_by_height=resized_height, show=True, wai_key=10)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
