import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np


class Stereoscopy:
    def __init__(self, max_shift):
        self.max_shift = max_shift

    def shift_image(self, image, depth_map):
        height, width = image.shape[:2]
        shifted_image = np.zeros((height, width, 3), dtype=np.uint8)

        depth_map_min = np.min(depth_map)
        depth_map_max = np.max(depth_map)

        for y in range(height):
            for x in range(width):
                # Calculate the normalized depth
                normalized_depth = (depth_map[y, x] - depth_map_min) / (depth_map_max - depth_map_min)

                # Calculate the shift amount (you can adjust the direction of shift here)
                shift = int(normalized_depth * self.max_shift)

                # Calculate new position
                new_x = x + shift

                # Ensure new position is within image bounds
                if 0 <= new_x < width:
                    shifted_image[y, new_x] = image[y, x]

        plt.imshow(shifted_image)
        plt.show()

        return shifted_image

    def create_mask(self, img):
        height, width = img.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        # Identify gaps in the offset image
        for y in range(height):
            for x in range(width):
                if np.all(img[y, x] == 0):
                    mask[y, x] = 255

        plt.imshow(mask, cmap='gray')
        plt.show()
        return mask

    def apply_inpainting(self, offset_image, mask, inpaint_radius):
        offset_image_uint8 = offset_image.astype(np.uint8)
        return cv2.inpaint(offset_image_uint8, mask, inpaint_radius, cv2.INPAINT_TELEA)

    def create_gif(self, image, shifted_image, gif_path):
        images = [image, shifted_image] * 10
        imageio.mimsave(gif_path, images, duration=0.25, loop=0)
