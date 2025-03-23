import os
import re
import cv2
import argparse
import numpy as np
from typing import List, Tuple
from collections import defaultdict
from PIL import Image
from lang_sam import LangSAM
from utils import nms
import warnings


def load_img_dict(path: str) -> dict:
    """Maps id to filename types for both formats."""
    img_dict = {}
    pattern1 = re.compile(r"^(.*?)_(\d+?)_(.*?)\.(jpg|jpeg|png)$")
    for filename in os.listdir(path):
        match1 = pattern1.match(filename)
        if match1:
            img, num, type, _ = match1.groups()
            key = f"{img}_{num}"
            if key not in img_dict:
                img_dict[key] = defaultdict(lambda: None)
            img_dict[key][type] = filename
    return img_dict

def load_prompts(file_path: str) -> List[str]:
    """Loads prompts from the given file and returns them as a list."""
    prompts = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                prompt = line.strip()
                if prompt:
                    prompts.append(prompt)
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {file_path}")
    return prompts

################### RAW Detected Patches + Bit Masks ######################## 

def run_langsam(img_dict: dict, prompts: List[str], directory: str) -> None:
    """Applies LangSAM to data"""
    model = LangSAM()
    for key, images in img_dict.items():
        img_raw = os.path.join(directory, images["raw"])
        img_pil_raw = Image.open(img_raw).convert("RGB")  
        prompt = ". ".join(prompts) + "."
        res = model.predict([img_pil_raw], [prompt])[0]
        res['labels'] = np.array(res['labels'])
        if len(res['labels']) == 0:
            warnings.warn(f"{key}: No object detected")
            continue
        filter_cls(res)  
        save_patch_raw(img_pil_raw, res['boxes'], res['labels'], key, directory)
        save_mask_raw(res['masks'], res['labels'], key, directory)
            
def filter_cls(res: dict):
    """Applies class based NMS filters to inputs"""
    # collect reduced results
    filtered_boxes = []
    filtered_labels = []
    filtered_logits = []
    filtered_masks = []
    # apply NMS to filtereach class
    for cls in np.unique(res['labels']):
        # unpack filtered classes
        cls_idx = res['labels'] == cls
        cls_box = res['boxes'][cls_idx]
        cls_logits = res['scores'][cls_idx]
        cls_label = res['labels'][cls_idx]
        cls_mask = res['masks'][cls_idx]
        # filter each class with nms
        boxes, labels, logits, masks = nms(cls_box, cls_label, cls_logits, cls_mask, 0.1)
        # accumulate remaining values
        filtered_boxes.append(boxes)
        filtered_labels.append(labels)
        filtered_logits.append(logits)
        filtered_masks.append(masks)
    # repack
    res['boxes'] = np.vstack(filtered_boxes) if filtered_boxes else np.array([])
    res['labels'] = np.hstack(filtered_labels) if filtered_labels else np.array([])
    res['scores'] = np.hstack(filtered_logits) if filtered_logits else np.array([])
    res['masks'] = np.vstack(filtered_masks) if filtered_masks else np.array([])

def save_patch_raw(img: np.array, boxes: np.array, labels: np.array, key: str, directory: str) -> None:
    """Uses specific bottom left, top right crop format btw"""
    # subdir
    outdir =  os.path.join(directory, 'patch_raw')
    os.makedirs(outdir, exist_ok=True)
    # counter
    label_id = defaultdict(int)
    # crop each box
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        label_id[labels[i]] += 1
        img_crop = img.crop((x1, y1, x2, y2))
        fname = f"{key}_cropped_{labels[i]}_{label_id[labels[i]]}.png"
        img_crop.save(os.path.join(outdir, fname))
             
def save_mask_raw(masks: np.array, labels: np.array, key: str, directory: str) -> None:
    """Applies mask with label for detected object and saves it"""
    # subdir
    outdir =  os.path.join(directory, 'mask_raw')
    os.makedirs(outdir, exist_ok=True)
    # counter
    label_id = defaultdict(int)
    # build mask
    for i, mask in enumerate(masks):
        label_id[labels[i]] += 1
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        fname = f"{key}_obstruction_real_mask_{labels[i]}_{label_id[labels[i]]}.png"
        mask_image.save(os.path.join(outdir, fname))

################### AR Patches ########################

def get_bounding_box(mask_path: str, tolerance: int=10) -> Tuple[int, int, int, int]:
    """"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    ys, xs = np.where(mask > 0)
    x_min, x_max = max(xs.min() - tolerance, 0), min(xs.max() + tolerance, mask.shape[1])
    y_min, y_max = max(ys.min() - tolerance, 0), min(ys.max() + tolerance, mask.shape[0])
    return (x_min, y_min, x_max - x_min, y_max - y_min)

def crop_bounding_box(image_path: str, bounding_box: Tuple[int, int, int, int]) -> np.ndarray:
    """"""
    image = cv2.imread(image_path)
    x_min, y_min, width, height = bounding_box
    x_max, y_max = x_min + width, y_min + height
    cropped_patch = image[y_min:y_max, x_min:x_max]
    return cropped_patch

def run_ar_patch(img_dict: dict, directory: str) -> None:
    """
    Executes the following steps per image
    1. Iterate through this for the full_virtual_mask.png
    2. Pass image_path to the get_bounding_box
    3. Pass the image_path and bounding box to the crop
    4. Pass the raw image_path and bounding box to the crop
    """
    # subdir
    dir_bg =  os.path.join(directory, 'patch_bg')
    dir_ar =  os.path.join(directory, 'patch_ar')
    os.makedirs(dir_bg, exist_ok=True)
    os.makedirs(dir_ar, exist_ok=True)
    # patches
    for key, images in img_dict.items():
        # images
        img_raw_path = os.path.join(directory, images["raw"])
        if "full_virtual_mask" in images:
            img_ar_path = os.path.join(directory, images["full_virtual_mask"])
        else:
            img_ar_path = os.path.join(directory, images["piece_virtual_mask"])
        # box
        bounding_box = get_bounding_box(img_ar_path, tolerance=10)
        # patches
        patch_raw = crop_bounding_box(img_raw_path, bounding_box)
        patch_ar = crop_bounding_box(img_ar_path, bounding_box)
        # write
        fname_bg = f"{key}_cropped_background.png"
        fname_ar = f"{key}_cropped_virtual.png"
        cv2.imwrite(os.path.join(dir_bg, fname_bg), patch_raw)
        cv2.imwrite(os.path.join(dir_ar, fname_ar), patch_ar)
        
# TESTS
            
def verify_img_dict(img_dict):
    for k, v in img_dict.items():
        if len(v.values()) < 4:
            warnings.warn(f"{k} missing an input img type, removing key")
            # del img_dict[k]


if __name__ == '__main__':
    # CLI arg parser
    parser = argparse.ArgumentParser(description="Process prompts from a file.")
    parser.add_argument(
        "-d", "--directory",
        type=str,
        default="data/ARQA_dataset/AVP/AVP_balloon/",
        help="Path to the prompt file (optional, defaults to default_prompts.txt)"
    )
    parser.add_argument(
        "-p", "--prompts",
        type=str,
        default="configs/prompts.txt",
        help="Path to the prompt file (optional, defaults to default_prompts.txt)"
    )
    # Load args
    args = parser.parse_args()
    target_directory = args.directory
    prompt_filepath = args.prompts
    # Load the data
    prompts = load_prompts(prompt_filepath)
    img_dict = load_img_dict(target_directory)
    verify_img_dict(img_dict)
    run_langsam(img_dict, prompts, target_directory)
    run_ar_patch(img_dict, target_directory)
