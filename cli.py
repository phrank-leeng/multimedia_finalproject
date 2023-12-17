import configparser
import matplotlib.pyplot as plt
from PIL import Image

from depth_map import DepthMapGenerator
from wiggle_stereoscopy import Stereoscopy

# main method
if __name__ == '__main__':
    # img_path = input("Enter the path to an image to generate a wiggly stereoscopy from:\n")
    # img_path = img_path.replace('"','')

    config_path = "config.ini"

    config = configparser.ConfigParser()
    config.read(config_path)

    image_path = config['ImagePaths']['image_path']
    depth_map_path = config['ImagePaths']['depth_map_path']
    output_gif_path = config['ImagePaths']['output_gif_path']
    shift_point = config['StereoscopySettings']['shift_point']

    dmg = DepthMapGenerator()
    img = dmg.load_img(image_path)
    depth_map = dmg.predict_depth_map(img)

    Image.fromarray(depth_map).save(depth_map_path)

    # img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # depth_map = cv2.imread("output.png", cv2.IMREAD_GRAYSCALE)

    stereo = Stereoscopy(config_path)
    shifted_img = stereo.shift_image(img, depth_map, shift_point)
    mask = stereo.create_mask(shifted_img)
    shifted_img = stereo.apply_inpainting(shifted_img, mask, 2)
    stereo.create_gif(img, shifted_img, output_gif_path, 1)

