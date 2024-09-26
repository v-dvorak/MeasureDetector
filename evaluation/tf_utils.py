from __future__ import annotations
import numpy as np
import tensorflow as tf
from PIL import Image
import json

def prepare_image(image_path):
    image = Image.open(image_path).convert("RGB") 
    return np.array(image), image.size

def retrieve_names_from_validations(path: str) -> list[str]:
    with open(path) as f:
        d = json.load(f)
    data = [x for xs in d["handwritten"].values() for x in xs]
    names = [x["path"].split("/")[-1] for x in data]
    return names

def prepare_ground_truth(gt_data, classes):
    """
    Input format: (top, left, bottom, right), absolute

    Ouptut format: (left, top, width, height), absolute
    """
    gt = []
    for i, current_class in enumerate(classes):
        for dat in gt_data[current_class]:
            gt.append(([dat["left"], dat["top"], dat["right"] - dat["left"], dat["bottom"] - dat["top"]], 1.0, i + 1))
    return gt

def prepare_prediction(prediction, width: int, height: int):
    """
    Input format: (top, left, bottom, right), relative

    Ouptut format: (left, top, width, height), absolute
    """
    pred = []
    for i in range(len(prediction["detection_boxes"])):
        # detection box coordinates format: (top, left, bottom, right), relative
        pred.append(
                    ([
                        prediction["detection_boxes"][i][1] * width, # left
                        prediction["detection_boxes"][i][0] * height, # top
                        (prediction["detection_boxes"][i][3] - prediction["detection_boxes"][i][1]) * width, # width
                        (prediction["detection_boxes"][i][2] - prediction["detection_boxes"][i][0]) * height, # height
                    ],
                    prediction["detection_scores"][i],
                    prediction["detection_classes"][i])
                    )
    return pred

def run_inference_for_single_image(image, graph):
    """
    Ruin interference for single image using a TF model.

    Author: A. Pacha
    """
    with graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections',
                'detection_boxes',
                'detection_scores',
                'detection_classes'
            ]:
                tensor_name = key + ':0'

                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]

            return output_dict
        
def load_detection_graph(path_to_checkpoint):
    """
    Loads TF model.

    Author: A. Pacha
    """
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_checkpoint, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph