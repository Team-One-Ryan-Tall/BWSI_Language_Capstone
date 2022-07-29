import random
from cogworks_data.language import get_data_path
from pathlib import Path
import json
import pickle as pkl
import ProcessText
from collections import Counter
from itertools import chain
import numpy as np
from gensim.models import KeyedVectors


class Database:
    def __init__(self) -> None:
        self.image_data = None
        self.caption_data = None
        
        filename = "glove.6B.200d.txt.w2v"
        print("loading GloVe")
        self.glove = KeyedVectors.load_word2vec_format(get_data_path(filename), binary=False)
        
        print("loading coco dataset")
        filename = get_data_path("captions_train2014.json")
        with Path(filename).open() as f:
            coco_data = json.load(f)
        
        print("loading resnet18")
        with Path(get_data_path('resnet18_features.pkl')).open('rb') as f:
            self.resnet18_features = pkl.load(f)
        
        print("Loading image url data")
        self.image_data = {image_info["id"] : (image_info["coco_url"], []) for image_info in coco_data["images"] if image_info["id"] in self.resnet18_features}
        
        print("Loading captions")
        self.caption_data = {caption_info["id"] : (caption_info["image_id"], caption_info["caption"]) for caption_info in coco_data["annotations"] if caption_info["image_id"] in self.resnet18_features}
        
        
        caption_list = self.append_image_caption_ids(coco_data)
        
        self.get_word_data(caption_list)
        
        print("generating caption embeddings")
        self.caption_embeddings = self.generate_caption_embeddings(list(self.caption_data.keys()))
        
        self.get_image_embeddings

    def append_image_caption_ids(self, coco_data):
        caption_list = []
        for caption in coco_data["annotations"]:
            image_id = caption["image_id"]
            caption_list.append(caption["caption"] + " ")
            if(image_id in self.resnet18_features):
                self.image_data[image_id][1].append(caption["id"])
        return caption_list


    def get_word_data(self, caption_list: "list[str]"):
        all_words = "".join(chain.from_iterable(caption_list))
        self.bag_of_words = ProcessText.create_bag_of_words(all_words, 10000)
        tokenized_captions = Counter(chain.from_iterable([set(ProcessText.tokenize(caption)) for caption in caption_list]))
        self.idfs = ProcessText.InverseFrequency(self.bag_of_words, tokenized_captions)
        print(self.idfs["a"], self.idfs["the"], self.idfs["woman"], self.idfs["traveling"])
        # print(self.bag_of_words[:20])
    
    def generate_caption_embeddings(self, caption_list: "list[int]"):
        return {caption : self.generate_caption_embedding(self.caption_data[caption][1]) for caption in caption_list}
    
    def generate_caption_embedding(self, caption):
        vec = np.array([self.glove[token.lower()] * self.idfs[token.lower()] if token in self.glove and self.idfs  else np.zeros(200) for token in ProcessText.tokenize(caption)])
        # print(vec.shape)
        vec = np.sum(vec, axis=0)
        vec /= np.linalg.norm(vec)
        return vec
    
    def get_features(self, image_id: int):
        return self.resnet18_features[image_id]
    
    def get_training_batch(self, batch_size: int):
        return [self.get_training_element() for _ in range(batch_size)]
    
    def get_training_element(self):
        true_image_id = self.get_random_image_id()
        confuser_image_id = self.get_random_image_id()
        true_caption_id = random.choice(self.image_data[true_image_id][1])
        return self.caption_embeddings[true_caption_id], self.resnet18_features[true_image_id], self.resnet18_features[confuser_image_id]
    
    def get_random_image_id(self):
        return random.choice(list(self.image_data.keys()))
    
    def get_captions(self, image_id: int):
        return [self.caption_data[caption_id][1] for caption_id in self.get_caption_ids(image_id)]
    
    def get_caption_ids(self, image_id: int):
        return  self.image_data[image_id][1]
    
    def generate_image_embeddings(self, model):
        self.image_embeddings = {image_id : model(self.resnet18_features[image_id]) for image_id in self.image_data}
    
def save_database(database:str, FileName: str): 
    with open(FileName, mode = "wb") as opened_file:
        pkl.dump(database, opened_file)

def load_database(FileName:str) -> Database:
    with open(FileName,"rb") as unopened_file:
        database = pkl.load(unopened_file)
    return database
    
# database = Database()
# save_database(database, "database.pkl")
