import numpy as np


def compute_iou(box1, box2):
    """Computes IoU between two bounding boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Compute intersection coordinates
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # Compute intersection area
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height

    # Compute areas of both boxes
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    # Compute IoU
    union_area = area1 + area2 - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def nms(boxes, labels, confidence, masks, iou_threshold=0.5):
    """
    Applies Non-Maximum Suppression (NMS) on bounding boxes.
    
    Parameters:
    - boxes: (N,4) NumPy array of bounding boxes (x1, y1, x2, y2)
    - labels: (N,) NumPy array of labels corresponding to each bounding box
    - iou_threshold: IoU threshold for suppression (default: 0.5)
    
    Returns:
    - filtered_boxes: (M,4) NumPy array of filtered bounding boxes
    - filtered_labels: (M,) NumPy array of corresponding labels
    """
    if len(boxes) == 0:
        return np.array([]), np.array([])
    # arrange data by box logit
    sorted_indices = np.argsort(confidence)[::-1]
    boxes = boxes[sorted_indices]
    labels = labels[sorted_indices]
    confidence = confidence[sorted_indices]
    masks = masks[sorted_indices]
    # collect data linked to useful box
    selected_boxes = []
    selected_labels = []
    selected_logits = []
    selected_masks = []
    # nms filter with IoU
    while len(boxes) > 0:
        # save top
        best_box = boxes[0]
        best_label = labels[0]
        best_logit = confidence[0]
        best_mask = masks[0]
        selected_boxes.append(best_box)
        selected_labels.append(best_label)
        selected_logits.append(best_logit)
        selected_masks.append(best_mask)
        # Compute IoU of the best box with the rest
        ious = np.array([compute_iou(best_box, box) for box in boxes[1:]])
        # Keep boxes with IoU below threshold
        keep_indices = np.where(ious < iou_threshold)[0] + 1  # +1 to shift index after best_box
        boxes = boxes[keep_indices]
        labels = labels[keep_indices]
        confidence = confidence[keep_indices]
        masks = masks[keep_indices]
    # conversions
    boxes = np.array(selected_boxes)
    labels = np.array(selected_labels)
    logits = np.array(selected_logits)
    masks = np.array(selected_masks)
    return boxes, labels, logits, masks
