import mygrad as mg
import mynn
from mynn.layers.dense import dense
from mygrad.nnet.losses.margin_ranking_loss import margin_ranking_loss
from mynn.optimizers.sgd import SGD
from mygrad.nnet.initializers import glorot_normal
from mygrad import no_autodiff
import pickle
from cogworks_data.language import get_data_path
from pathlib import Path
import Database
import numpy as np
import pickle as pkl


class Autoencoder:
    def __init__(self, D_full=None, D_hidden=None, generate=True):
        if(generate):
            self.encode = dense(D_full, D_hidden, weight_initializer=glorot_normal, bias=False)
        else:
            self.encode = dense(0,0)
    
    def __call__(self, x):        
        # try:
        #TODO: NORMALIZE
        #x = (N,512) ndarray
        #out = (N,200) Tensor
        # print(x.shape)
        out = self.encode(x)
        #for each N vector compute norms and divide vector by norms
        out /= mg.linalg.norm(out, axis=1, keepdims=True)
        return out 
        # except:
        #     print("Exception")
    
    
    @property
    def parameters(self):
        return self.encode.parameters

def unzip(pairs):
    return tuple(zip(*pairs))

def loss_function_accuracy(wtruecap, wtrueimg, wconfimg, margin): 
    # print(wtrueimg.dtype, wtruecap.dtype, wconfimg.dtype)
    simtrue = mg.einsum("ni,ni -> n", wtruecap, wtrueimg)
    simconfuser = mg.einsum("ni,ni -> n", wtruecap, wconfimg)

    loss = margin_ranking_loss(simtrue, simconfuser, 1, margin=0.25)
    #returns both loss and accuracy, loss first
    
    accuracy = np.mean(simtrue.data  > simconfuser.data)  
    #mg.mean(mg.maximum(0, margin - (simtrue - simconfuser)))
    return (loss, accuracy)

def train_model(model: Autoencoder, database: Database.Database, batches=5000, batch_size=32, epochs=100, optim=SGD, learning_rate=1e-3, plotter=None):
    optim = SGD(model.parameters, learning_rate=learning_rate, momentum=0.9) 
    # master = database.get_training_batch(batch_size)
    for _ in range(epochs):
        for _ in range(batch_size):
            triplets = database.get_training_batch(batch_size)
            # triplets = master
            caption_embeds, trues_embeds, confusers_embeds = unzip(triplets)
            caption_embeds = np.array(list(caption_embeds))
            trues_embeds = np.array(list(trues_embeds))
            confusers_embeds = np.array(list(confusers_embeds))
            trues_embeds = model(trues_embeds)
            confusers_embeds = model(confusers_embeds)
            # triplets = [(caption_id, model(true_id), model(confusor_id)) for caption_id, true_id, confusor_id in triplets]
            loss, acc = loss_function_accuracy(caption_embeds, trues_embeds, confusers_embeds, 0.25)
            loss.backward()
            # print(loss.grad)
            optim.step()
            if plotter is not None:
                plotter.set_train_batch({"loss" : loss.item(),
                                        "accuracy" : acc},
                                        batch_size=batch_size)
                mg.turn_memory_guarding_off()
        # print(optim.params[0].grad)
        if plotter is not None:
                plotter.set_train_epoch()
        print(loss.item(), acc)
        plotter.plot()
    
def save_model(model: Autoencoder, FileName: str): 
    print("Saving Model")
    with open(FileName, mode = "wb") as opened_file:
        np.savez(opened_file, *(x.data for x in model.parameters))
    print("Done!")
    
def load_model(FileName:str) -> Database:
    print("Loading Model")
    with open(FileName,"rb") as unopened_file:
        model = Autoencoder(D_full=512, D_hidden=200)
        for param, (name, array) in zip(model.parameters, np.load(unopened_file).items()):
            param.data[:] = array
    print("Done!")
    return model

# database = Database.Database()
# Database.save_database(database, "database.pkl")

# database = Database.load_database("database.pkl")
# model = Autoencoder(512, 200)

# train_model(model, database)