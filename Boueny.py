import random
from cogworks_data.language import get_data_path
from pathlib import Path
import json
import pickle

import numpy as np

class CocoDatabase:
    def __init__(self) -> None:
        self.image_data = None
        self.caption_data = None
        with Path(get_data_path('resnet18_features.pkl')).open('rb') as f:
            self.resnet18_features = pickle.load(f)
        # self.image_embeddings = #TODO GENERATE IMAGE EMBEDDINGS
        # self.caption_embeddings = #TODO GENERATE CAPTION EMBEDDINGS
        
        filename = get_data_path("captions_train2014.json")
        with Path(filename).open() as f:
            coco_data = json.load(f)
        # print(len(coco_data["images"]), coco_data["images"][10])
        print("Loading image url data")
        self.image_data = {image_info["id"] : (image_info["coco_url"], []) for image_info in coco_data["images"] if image_info["id"] in self.resnet18_features}
        print("Loading captions")
        self.caption_data = {caption_info["id"] : (caption_info["image_id"], caption_info["caption"]) for caption_info in coco_data["annotations"] if caption_info["image_id"] in self.resnet18_features}
        # print(list(self.image_data.items())[0])
        for caption in coco_data["annotations"]:
            image_id = caption["image_id"]
            if(image_id in self.resnet18_features):
                self.image_data[image_id][1].append(caption["id"])
        # print(list(self.image_data.items())[0])
        # print(self.image_data[42493])
        # print(self.get_random_image_id())
        # print(self.get_captions(self.get_random_image_id()))
        print(self.get_training_batch(200))
        
    def get_features(self, image_id: int):
        return self.resnet18_features[image_id]
    def get_training_batch(self, batch_size: int):
        return [self.get_training_element() for _ in range(batch_size)]
    def get_training_element(self):
        true_image_id = self.get_random_image_id()
        confuser_image_id = self.get_random_image_id()
        true_caption_id = random.choice(self.image_data[true_image_id][1])
        return true_caption_id, true_image_id, confuser_image_id
    def get_random_image_id(self):
        return random.choice(list(self.image_data.keys()))
    def get_captions(self, image_id: int):
        return [self.caption_data[caption_id][1] for caption_id in self.get_caption_ids(image_id)]
    def get_caption_ids(self, image_id: int):
        return  self.image_data[image_id][1]
coco = CocoDatabase()