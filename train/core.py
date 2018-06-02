import numpy as np

import batch
import builder

from keras.callbacks import ModelCheckpoint
from keras.models import load_model

""" Custom imports """
from visualization import tools
import areaBuilder
from models import uNet
from models import model_tools
from region import objArea

seed = 7
np.random.seed(seed)
BATCH_SIZE=2

def test(index=0):
    # Load uNet model first
    model_u = load_model('logs/unet.hdf5', custom_objects={'focal_loss_fixed': uNet.focal_loss(gamma=2., alpha=.6), 'f1': uNet.f1})
    print('uNet model is loading...')
    print(model_u.summary())
    
    # Train and val pickle opened
    a_train = areaBuilder.Area(model_u, split='train')
    a_val = areaBuilder.Area(model_u, split='val')
    
    LEN_TRAIN_DATASET = len(a_train)
    LEN_VAL_DATASET = len(a_val)
    
    train_idxs = np.arange(0, LEN_TRAIN_DATASET)
    np.random.shuffle(train_idxs)
    
    val_idxs = np.arange(0, LEN_VAL_DATASET)
    np.random.shuffle(val_idxs)
    
    train_generator = batch.Batch(train_idxs, a_train, batch_size=BATCH_SIZE, split='train', bbox_train=True)
    val_generator = batch.Batch(val_idxs, a_val, batch_size=BATCH_SIZE, split='val', bbox_train=True)
        
    # Add a checkpoint for our model
    model_checkpoint = ModelCheckpoint('logs/bbNet.hdf5', monitor='loss', verbose=1, save_best_only=True)
    # save weights
    sw = model_tools.WeightsSaver(model)
    # plot the losses values 
    pl = model_tools.PlotLosses(20)
    
    results = model.fit_generator(
                generator=train_generator.__start__(int(LEN_TRAIN_DATASET/BATCH_SIZE) - 1),
                validation_data=val_generator.__start__(int(LEN_VAL_DATASET/BATCH_SIZE) - 1),
                steps_per_epoch=int(LEN_TRAIN_DATASET/BATCH_SIZE),
                validation_steps=int(LEN_VAL_DATASET/BATCH_SIZE),
                epochs=20,
                callbacks=[model_checkpoint, sw, pl])
    
if __name__=='__main__':
    import sys
    if len(sys.argv) > 1:
        test(index=int(sys.argv[1]))
    else:
        test()