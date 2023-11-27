import cv2
import torch

import matplotlib.pyplot as plt

class WiggleStereoscopy:
    def __init__(self):
        model_type = "DPT_Large"  # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
        # model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
        # model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

        midas = torch.hub.load("intel-isl/MiDaS", model_type)

        # move model to GPU if available
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        midas.to(device)
        midas.eval()

        # load transforms to resize and normalize the image for large or small model
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

        def load_img(path):
            img = cv2.imread(path)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        def predict_depth_map(img):
            input_batch = self.transform(img).to(device)

            with torch.no_grad():
                prediction = midas(input_batch)

                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            output = prediction.cpu().numpy()

            plt.imshow(output)


