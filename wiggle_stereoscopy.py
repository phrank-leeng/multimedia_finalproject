import configparser
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np


class Stereoscopy:
    def __init__(self, config_path):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.max_shift = int(self.config['StereoscopySettings']['max_shift'])
        self.frame_duration = float(self.config['StereoscopySettings']['frame_duration'])
        self.repetitions = int(self.config['StereoscopySettings']['repetitions'])
        self.shift_point = int(self.config['StereoscopySettings']['shift_point'])


    def shift_image(self, image, depth_map, shift_point):
        print("Shifting image...")
        height, width = image.shape[:2]
        shifted_image = np.zeros((height, width, 3), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                depth_value = depth_map[y, x]

                # Determine shift direction and magnitude
                if depth_value < shift_point:
                    # Closer objects - shift to the left
                    shift = int(((shift_point - depth_value) / shift_point) * self.max_shift)
                    new_x = x - shift
                else:
                    # Further objects - shift to the right
                    shift = int(((depth_value - shift_point) / (255 - shift_point)) * self.max_shift)
                    new_x = x + shift

                # Ensure new position is within image bounds
                if 0 <= new_x < width:
                    shifted_image[y, new_x] = image[y, x]
        return shifted_image

    def create_mask(self, img):
        print("Creating mask...")
        height, width = img.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        # Identify gaps in the offset image
        for y in range(height):
            for x in range(width):
                if np.all(img[y, x] == 0):
                    mask[y, x] = 255
        return mask

    def apply_inpainting(self, offset_image, mask, inpaint_radius):
        print("Applying inpainting...")
        offset_image_uint8 = offset_image.astype(np.uint8)
        return cv2.inpaint(offset_image_uint8, mask, inpaint_radius, cv2.INPAINT_TELEA)

    def create_gif(self, image, shifted_image, gif_path, compression_ratio):
        print("Creating GIF...")
        images = [image, shifted_image] * self.repetitions
        resized_images = [cv2.resize(img, (img.shape[1]//compression_ratio, img.shape[0]//compression_ratio)) for img in images]
        imageio.mimsave(gif_path, resized_images, duration=self.frame_duration, loop=0)
