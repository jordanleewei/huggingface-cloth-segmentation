# process.py

import os
import base64
import requests
from io import BytesIO
import pymongo

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
from collections import OrderedDict

# 1. Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

# ---------------------------
#  Imports from your modules
# ---------------------------
from network import U2NET  # Ensure this is the correct path to your U2NET definition
# from options import opt     # If you have a separate options.py with an opt object, uncomment this line

# ---------------------------------------------------------------------
# If you don't have an external options.py, you can define 'opt' here:
# ---------------------------------------------------------------------
class OptPlaceholder:
    output = 'results'  # path to store intermediate outputs (masks, seg, etc.) locally
opt = OptPlaceholder()

# --------------------------------------------------
#  U^2-Net Loading & Utilities (your original code)
# --------------------------------------------------

def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print("----No checkpoints at given path----")
        return
    model_state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:]  # remove `module.` from state_dict keys if needed
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    print("----checkpoints loaded from path: {}----".format(checkpoint_path))
    return model


def get_palette(num_cls):
    """Returns the color map for visualizing the segmentation mask."""
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


class Normalize_image(object):
    """Normalize given tensor into given mean and standard deviation."""

    def __init__(self, mean, std):
        assert isinstance(mean, float)
        self.mean = mean
        self.std = std

        self.normalize_1 = transforms.Normalize(self.mean, self.std)
        self.normalize_3 = transforms.Normalize([self.mean]*3, [self.std]*3)
        self.normalize_18 = transforms.Normalize([self.mean]*18, [self.std]*18)

    def __call__(self, image_tensor):
        if image_tensor.shape[0] == 1:
            return self.normalize_1(image_tensor)
        elif image_tensor.shape[0] == 3:
            return self.normalize_3(image_tensor)
        elif image_tensor.shape[0] == 18:
            return self.normalize_18(image_tensor)
        else:
            raise ValueError("Normalization implemented only for 1, 3, or 18-channel images.")


def apply_transform(img):
    transforms_list = [transforms.ToTensor(), Normalize_image(0.5, 0.5)]
    transform_rgb = transforms.Compose(transforms_list)
    return transform_rgb(img)


def generate_mask(input_image, net, palette, device='cpu'):
    """
    Performs cloth segmentation on input_image with the given U^2-Net model,
    and returns a PIL 'P' mode image representing the segmentation map.
    """
    # Prepare directories for alpha and cloth_seg output (if you want local saving)
    alpha_out_dir = os.path.join(opt.output, 'alpha')
    cloth_seg_out_dir = os.path.join(opt.output, 'cloth_seg')
    os.makedirs(alpha_out_dir, exist_ok=True)
    os.makedirs(cloth_seg_out_dir, exist_ok=True)

    # Resize to match model input
    orig_size = input_image.size
    resized_img = input_image.resize((768, 768), Image.BICUBIC)
    image_tensor = apply_transform(resized_img)
    image_tensor = torch.unsqueeze(image_tensor, 0).to(device)

    with torch.no_grad():
        output_tensor = net(image_tensor)
        # U^2-Net typically returns multiple outputs in a list; here we assume [0] is the final
        output_tensor = F.log_softmax(output_tensor[0], dim=1)
        output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
        # Squeeze the batch dimension => shape might become [1, H, W]
        output_tensor = torch.squeeze(output_tensor, dim=0)

        output_arr = output_tensor.cpu().numpy()

    # --------------------------------------------------------------------
    # FIX: Make sure output_arr is 2D [H, W], not 3D [1, H, W].
    # --------------------------------------------------------------------
    if output_arr.ndim == 3 and output_arr.shape[0] == 1:
        output_arr = output_arr[0]

    # Save alpha masks for classes 1, 2, 3 (top, bottom, possibly other garment)
    for cls in range(1, 4):
        if np.any(output_arr == cls):
            alpha_mask = (output_arr == cls).astype(np.uint8) * 255
            alpha_mask_img = Image.fromarray(alpha_mask, mode='L')
            # Resize back to original
            alpha_mask_img = alpha_mask_img.resize(orig_size, Image.BICUBIC)
            alpha_mask_img.save(os.path.join(alpha_out_dir, f'{cls}.png'))

    # Create final segmentation image (P mode with palette)
    cloth_seg = Image.fromarray(output_arr.astype(np.uint8), mode='P')
    cloth_seg.putpalette(palette)
    cloth_seg = cloth_seg.resize(orig_size, Image.BICUBIC)
    cloth_seg.save(os.path.join(cloth_seg_out_dir, 'final_seg.png'))

    return cloth_seg


def get_top_bottom_in_memory(original_image, segmentation_image,
                             top_class=1, bottom_class=2):
    """
    Returns two PIL images (top_cropped, bottom_cropped) in memory instead of saving to disk.
    If a part doesn't exist, returns None for that image.
    """
    seg_array = np.array(segmentation_image)

    top_cropped_img = None
    bottom_cropped_img = None

    # ---------------------------
    # Top Crop
    # ---------------------------
    if (seg_array == top_class).any():
        top_mask = (seg_array == top_class).astype(np.uint8) * 255
        top_mask_pil = Image.fromarray(top_mask, mode='L')
        top_crop = original_image.convert('RGBA')
        top_crop.putalpha(top_mask_pil)

        alpha_channel = top_crop.split()[3]
        bbox = alpha_channel.getbbox()
        if bbox:
            top_crop = top_crop.crop(bbox)
        top_cropped_img = top_crop

    # ---------------------------
    # Bottom Crop
    # ---------------------------
    if (seg_array == bottom_class).any():
        bottom_mask = (seg_array == bottom_class).astype(np.uint8) * 255
        bottom_mask_pil = Image.fromarray(bottom_mask, mode='L')
        bottom_crop = original_image.convert('RGBA')
        bottom_crop.putalpha(bottom_mask_pil)

        alpha_channel = bottom_crop.split()[3]
        bbox = alpha_channel.getbbox()
        if bbox:
            bottom_crop = bottom_crop.crop(bbox)
        bottom_cropped_img = bottom_crop

    return top_cropped_img, bottom_cropped_img


def check_or_download_model(file_path):
    """
    If you want to automatically download the model if not present,
    you can integrate gdown or your own method here.
    """
    if not os.path.exists(file_path):
        print(f"Model not found at {file_path}. Please download or place it manually.")
    else:
        print("Model already exists.")


def load_seg_model(checkpoint_path, device='cpu'):
    net = U2NET(in_ch=3, out_ch=4)
    check_or_download_model(checkpoint_path)
    net = load_checkpoint(net, checkpoint_path)
    net = net.to(device)
    net.eval()
    return net


# --------------------------------
# MAIN BATCH PROCESSING ENTRYPOINT
# --------------------------------

def main():
    # ---------------------
    # 1. Connect to MongoDB
    # ---------------------
    # Read the MongoDB connection string from environment
    mongo_uri = os.getenv("MONGO_URI")  # or "MONGODB_CONNECTION_STRING" if you prefer

    if not mongo_uri:
        raise ValueError("MONGO_URI not found in environment variables. "
                         "Please set it in the .env file.")

    client = pymongo.MongoClient(mongo_uri)
    db = client["kagame"]        # Adjust to your database name
    collection = db["catalogue"] # Adjust to your collection name

    # ---------------------
    # 2. Load U^2-Net Model
    # ---------------------
    device = 'cpu'  # or 'cuda:0' if you have GPU
    checkpoint_path = 'model/cloth_segm.pth'  # Adjust path if needed
    model = load_seg_model(checkpoint_path, device=device)
    palette = get_palette(4)

    # -------------------------------------------------
    # 3. Iterate Over Catalogue Items & Process Each
    # -------------------------------------------------
    items = collection.find({})
    for item in items:
        category = item.get("category", "")
        image_url = item.get("image_url", "")

        # We only process items that have an image_url and are in "Tops" or "Bottoms"
        if not image_url or category not in ["Tops", "Bottoms"]:
            continue

        print(f"Processing item _id={item['_id']} with category={category}")

        try:
            # 3.1 Download the image
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")

            # 3.2 Run segmentation to get the cloth mask
            cloth_seg = generate_mask(img, net=model, palette=palette, device=device)

            # 3.3 Get top & bottom cropped images in memory
            top_cropped, bottom_cropped = get_top_bottom_in_memory(
                original_image=img,
                segmentation_image=cloth_seg,
                top_class=1,
                bottom_class=2
            )

            # 3.4 Depending on the category, store the appropriate cropped image in DB
            cropped_image_b64 = None
            if category == "Tops" and top_cropped is not None:
                buf = BytesIO()
                top_cropped.save(buf, format="PNG")
                cropped_image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            elif category == "Bottoms" and bottom_cropped is not None:
                buf = BytesIO()
                bottom_cropped.save(buf, format="PNG")
                cropped_image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            # 3.5 Update the document in MongoDB if we actually have a cropped image
            if cropped_image_b64:
                collection.update_one(
                    {"_id": item["_id"]},
                    {"$set": {"cropped_image": cropped_image_b64}}
                )
                print(f"Updated item {item['_id']} with cropped_image.")
            else:
                print(f"No cropped image available for item {item['_id']}.")

        except Exception as e:
            print(f"Error processing item {item['_id']}: {e}")


if __name__ == '__main__':
    main()
