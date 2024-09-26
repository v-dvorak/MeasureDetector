from __future__ import annotations
from glob import glob
import random
import argparse
import json
from tqdm import tqdm
import json
import os

# import tf_utils
import yolo_utils
import eval_utils

CLASSES = ["system_measures", "stave_measures", "staves", "systems", "grand_staff"]

def yolo_pred_gt_job(model_path: str, val_dir: str, count: int, seed: int=42) -> tuple[
    list[tuple[list[float, float, float, float], float, int]],
    list[tuple[list[float, float, float, float], float, int]]]:

    MODEL = yolo_utils.load_model(model_path)
    images = list(glob(val_dir + "/*.png"))

    if count is not None:
        random.Random(seed).shuffle(images)
        images = images[:count]
    
    GROUND_TRUTH = []
    PREDICTIONS = []

    for image in tqdm(images):
        im, (width, height) = yolo_utils.prepare_image(image)

        prediction = MODEL.predict(source=im)[0]
        
        PREDICTIONS.append(yolo_utils.prepare_prediction(prediction.boxes))
        GROUND_TRUTH.append(yolo_utils.prepare_ground_truth(yolo_utils.get_gt_path(image), width, height))

        # eval_utils.draw_rectangles_with_conf(image, yolo_utils.prepare_prediction(prediction.boxes), 5)
    
    return GROUND_TRUTH, PREDICTIONS

def tensorflow_pred_gt_job(model_path: str, val_dir: str, count: int, seed: int=42, val_file:str = None) -> tuple[
    list[tuple[list[float, float, float, float], float, int]],
    list[tuple[list[float, float, float, float], float, int]]]:
    """
    Retrieves ground truth and prediction using given model and evaluation dataset.
    """
    MODEL = tf_utils.load_detection_graph(model_path)

    if val_file is None:
        images = list(glob(val_dir + "/*.png"))
    else:
        images = tf_utils.retrieve_names_from_validations(val_file)
        images = [os.path.join(val_dir, img) for img in images]

    if count is not None:
        random.Random(seed).shuffle(images)
        images = images[:count]
    
    GROUND_TRUTH = []
    PREDICTIONS = []

    for image in tqdm(images):
        image_np, (width, height) = tf_utils.prepare_image(image)
        
        with open(image[:-4] + ".json", "r", encoding="utf8") as f:
            data_gt = json.load(f)
        
        annot_gt = tf_utils.prepare_ground_truth(data_gt, CLASSES)
        prediction = tf_utils.run_inference_for_single_image(image_np, MODEL)
        pred_prep = tf_utils.prepare_prediction(prediction, width, height)

        GROUND_TRUTH.append(annot_gt)
        PREDICTIONS.append(pred_prep)

        # eval_utils.draw_rectangles_with_conf(image, pred_prep, 5, threshold=0.05)
    
    return GROUND_TRUTH, PREDICTIONS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TensorFlow object detection model on a validation dataset.")
    parser.add_argument("model_path", type=str, help="Path to model.")
    parser.add_argument("dataset_dir", type=str, help="Path to validation dataset.")
    parser.add_argument("-v", "--val_file", type=str, help="Path to JSON with validation data.")
    parser.add_argument("-c", "--count", type=int, help="How many images the model will be tested on.")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Seed for dataset shuffling.")
    args = parser.parse_args()

    GROUND_TRUTH, PREDICTIONS = yolo_pred_gt_job(args.model_path, args.dataset_dir, args.count, seed=int(args.seed))
    # GROUND_TRUTH, PREDICTIONS = tensorflow_pred_gt_job(args.model_path, args.dataset_dir, args.count, seed=int(args.seed), val_file=args.val_file)

    eval_utils.evaluate_metrics(GROUND_TRUTH, PREDICTIONS, CLASSES)
