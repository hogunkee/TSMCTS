import os
import cv2
import numpy as np
import supervision as sv

import torch
import torchvision

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor


class GroundedSAM:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # GroundingDINO config and checkpoint
    GSAM_PATH = "/home/ur-plusle/Desktop/Grounded-Segment-Anything"
    GROUNDING_DINO_CONFIG_PATH = os.path.join(GSAM_PATH, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(GSAM_PATH, "./groundingdino_swint_ogc.pth")

    # Segment-Anything checkpoint
    SAM_ENCODER_VERSION = "vit_h"
    SAM_CHECKPOINT_PATH = os.path.join(GSAM_PATH, "./sam_vit_h_4b8939.pth")

    def __init__(self, gsam_path=None):
        if gsam_path is not None:
            self.GSAM_PATH = gsam_path
            self.GROUNDING_DINO_CONFIG_PATH = os.path.join(gsam_path, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
            self.GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(gsam_path, "./groundingdino_swint_ogc.pth")
            self.SAM_CHECKPOINT_PATH = os.path.join(gsam_path, "./sam_vit_h_4b8939.pth")

        # Building GroundingDINO inference model
        self.grounding_dino_model = Model(
                model_config_path=self.GROUNDING_DINO_CONFIG_PATH, 
                model_checkpoint_path=self.GROUNDING_DINO_CHECKPOINT_PATH)

        # Building SAM Model and SAM Predictor
        self.sam = sam_model_registry[self.SAM_ENCODER_VERSION](checkpoint=self.SAM_CHECKPOINT_PATH)
        self.sam.to(device=self.DEVICE)
        self.sam_predictor = SamPredictor(self.sam)

    def get_masks(self, image, classes, box_threshold=0.25, text_threshold=0.25, nms_threshold=0.8, save_image=False):
        #CLASSES = ["The running dog"]
        #BOX_THRESHOLD = 0.25
        #TEXT_THRESHOLD = 0.25
        #NMS_THRESHOLD = 0.8

        # detect objects
        detections = self.grounding_dino_model.predict_with_classes(
            image=image,
            classes=classes,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )

        if save_image:
            # annotate image with detections
            box_annotator = sv.BoxAnnotator()
            labels = [
                f"{classes[class_id]} {confidence:0.2f}" 
                for _, _, confidence, class_id, _, _ 
                in detections]
            annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

            # save the annotated grounding dino image
            cv2.imwrite("outputs/groundingdino_annotated_image.jpg", annotated_frame)


        # NMS post process
        print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            nms_threshold
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        print(f"After NMS: {len(detections.xyxy)} boxes")

        # convert detections to masks
        detections.mask = self.segment(
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )

        if save_image:
            # annotate image with detections
            box_annotator = sv.BoxAnnotator()
            mask_annotator = sv.MaskAnnotator()
            labels = [
                f"{classes[class_id]} {confidence:0.2f}" 
                for _, _, confidence, class_id, _, _ 
                in detections]
            annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
            annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

            # save the annotated grounded-sam image
            cv2.imwrite("outputs/grounded_sam_annotated_image.jpg", annotated_image)

        return detections

    # Prompting SAM with detected boxes
    def segment(self, image, xyxy):
        self.sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = self.sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)


