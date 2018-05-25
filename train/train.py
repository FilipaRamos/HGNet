import numpy as np

""" Custom imports """
import batch
import builder
from models import segNet

seed = 7
np.random.seed(seed)
BATCH_SIZE = 32
NUM_POINTS = 1048

def train():    
    # Train and val pickle opened
    fr_train = builder.Frustum(split='train', num_points=NUM_POINTS)
    fr_val = builder.Frustum(split='val', num_points=NUM_POINTS)
    
    LEN_TRAIN_DATASET = len(fr_train)
    LEN_VAL_DATASET = len(fr_val)
    
    train_idxs = np.arange(0, LEN_TRAIN_DATASET)
    np.random.shuffle(train_idxs)
    
    val_idxs = np.arange(0, LEN_VAL_DATASET)
    np.random.shuffle(val_idxs)
    
    train_generator = batch.Batch(train_idxs, fr_train, batch_size=BATCH_SIZE, split='train')
    val_generator = batch.Batch(val_idxs, fr_val, batch_size=BATCH_SIZE, split='val')
    
    sg = segNet.SegNet(batch_size=BATCH_SIZE)
    model = sg.model
    print(model.summary())
    model.fit_generator(
                generator=train_generator,
                validation_data=val_generator,
                steps_per_epoch=int(LEN_TRAIN_DATASET/BATCH_SIZE),
                epochs=150)
    
    
if __name__=='__main__':
    train()