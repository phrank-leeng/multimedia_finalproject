import cv2
import matplotlib.pyplot as plt

from depth_map import DepthMapGenerator
from wiggle_stereoscopy import Stereoscopy

# main method
if __name__ == '__main__':
    # img_path = input("Enter the path to an image to generate a wiggly stereoscopy from:\n")
    # img_path = img_path.replace('"','')
    img_path = "C:\\Users\\Frank\\Pictures\\58eyvu.png"
    # dmg = DepthMapGenerator()
    # img = dmg.load_img(img_path)
    # depth_map = dmg.predict_depth_map(img)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    depth_map = cv2.imread("output.png", cv2.IMREAD_GRAYSCALE)

    plt.imshow(img)
    plt.show()

    stereo = Stereoscopy(25)
    shifted_img = stereo.shift_image(img, depth_map)
    mask = stereo.create_mask(shifted_img)
    shifted_img = stereo.apply_inpainting(shifted_img, mask, 3)
    stereo.create_gif(img, shifted_img, "wiggle.gif")

