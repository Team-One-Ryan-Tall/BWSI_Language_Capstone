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
import mygrad as mg
from IPython.display import Image, display


class Database:
    def __init__(self, generate=True) -> None:
        if generate:
            self.image_embeddings = None
            
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
        doc_count = len(caption_list)
        self.idfs = ProcessText.InverseFrequency(tokenized_captions, doc_count)
        # print(self.idfs["a"], self.idfs["the"], self.idfs["woman"], self.idfs["traveling"])
        # print(self.bag_of_words[:20])
    
    def generate_caption_embeddings(self, caption_list: "list[int]"):
        return {caption : self.generate_caption_embedding(self.caption_data[caption][1]) for caption in caption_list}
    
    def generate_caption_embedding(self, caption):
        vec = np.array([self.glove[token.lower()] * self.idfs[token.lower()] if token in self.glove and self.idfs  else np.zeros(200) for token in ProcessText.tokenize(caption)])
        # print(vec.shape)
        vec = np.sum(vec, axis=0)
        vec /= np.linalg.norm(vec)
        return vec
    
    def get_random_caption_embedding(self):
        caption_id =  random.choice(list(self.caption_data.keys()))
        return (caption_id, self.caption_data[caption_id][1])
    
    def get_features(self, image_id: int):
        return self.resnet18_features[image_id]
    
    def get_training_batch(self, batch_size: int):
        return [self.get_training_element() for _ in range(batch_size)]
    
    def get_training_element(self):
        true_image_id = self.get_random_image_id()
        confuser_image_id = self.get_random_image_id()
        true_caption_id = random.choice(self.image_data[true_image_id][1])
        return self.caption_embeddings[true_caption_id], self.resnet18_features[true_image_id].flatten(), self.resnet18_features[confuser_image_id].flatten()
    
    def get_random_image_id(self):
        return random.choice(list(self.image_data.keys()))
    
    def get_captions(self, image_id: int):
        return [self.caption_data[caption_id][1] for caption_id in self.get_caption_ids(image_id)]
    
    def get_caption_ids(self, image_id: int):
        return  self.image_data[image_id][1]
    
    def generate_image_embeddings(self, model):
        with mg.no_autodiff:
            self.image_embeddings = {image_id : model(self.resnet18_features[image_id]) for image_id in self.image_data}
    
    
    def query_database(self, caption_embedding, k): 
        image_similarities = {(self.image_embeddings[image_id] @ caption_embedding).item() : image_id for image_id in self.image_embeddings}

        image_similarities = [image_similarities[i] for i in  reversed(sorted(image_similarities.keys()))]
        
        k_closest_embeddings = image_similarities[0:k]
        
        return [self.image_data[image_id][0] for image_id in k_closest_embeddings]
        # values = []
        # for i in image_similarities:
        #     for j in k_closest_embeddings:
        #         if j == i[0]:
        #             values.append(i[1])
                    
        # final_embeddings = []
        # for x in values:
        #     final_embeddings.append(list_of_caption_embeddings.index(x)) #replace with database of caption embeddings
        
        # return final_embeddings #still need function that takes in caption embeddings and returns images
    
def url_to_image(image_urls): 
    for _ in image_urls:
        display(Image(url = _))

def save_database(database: Database, FileName: str): 
    print("Saving Database")
    with open(FileName, mode = "wb") as opened_file:
        pickle = (database.glove, database.resnet18_features, database.bag_of_words, database.image_data, database.caption_data, database.image_embeddings, database.caption_embeddings, database.idfs)
        pkl.dump(pickle, opened_file)
    print("Done!")

def load_database(FileName:str) -> Database:
    print("Loading Database")
    with open(FileName,"rb") as unopened_file:
        unpickle = pkl.load(unopened_file)
    database = Database(False)
    database.glove, database.resnet18_features, database.bag_of_words, database.image_data, database.caption_data, database.image_embeddings, database.caption_embeddings, database.idfs = unpickle
    print("Done!")
    return database
# database = load_database("database2.pkl")
# rand = database.get_random_caption_embedding()
# print(rand[1])
# print(rand[0])
# database.query_database(database.caption_embeddings[rand[0]], 10)
# # database = Database()
# # save_database(database, "database.pkl")
# # database = load_database("database.pkl")
# # print("")