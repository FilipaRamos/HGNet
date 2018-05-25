import numpy as np

""" Custom imports """
import batch
from models import segNet

seed = 7
np.random.seed(seed)
LEN_TRAIN_DATASET = 73508
LEN_EVAL_DATASET = 35000
BATCH_SIZE = 32

def train():
    train_idxs = np.arange(0, LEN_TRAIN_DATASET)
    np.random.shuffle(train_idxs)
    num_batches = LEN_TRAIN_DATASET/BATCH_SIZE
    
    eval_idxs = np.arange(0, LEN_EVAL_DATASET)
    np.random.shuffle(eval_idxs)
    num_batches = LEN_EVAL_DATASET/BATCH_SIZE
    
    training_generator = batch.Batch(train_idxs, batch_size=BATCH_SIZE, split='train')
    validation_generator = batch.Batch(eval_idxs, batch_size=BATCH_SIZE, split='val')
    
    sg = segNet.SegNet(batch_size=BATCH_SIZE)
    model = sg.model
    model.fit_generator(
                train_generator,
                steps_per_epoch=LEN_TRAIN_DATASET/BATCH_SIZE,
                epochs=20,
                validation_data=validation_generator)
    
    
if __name__=='__main__':
    train()