import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

""" Custom imports """
import batch
import builder
from models import segNet
from models import uNet
from models import model_tools

seed = 7
np.random.seed(seed)
BATCH_SIZE = 64

def train(LOAD_FLAG=False):    
    # Train and val pickle opened
    fr_train = builder.Frustum(split='train')
    fr_val = builder.Frustum(split='val')
    
    LEN_TRAIN_DATASET = len(fr_train)
    LEN_VAL_DATASET = len(fr_val)
    
    train_idxs = np.arange(0, LEN_TRAIN_DATASET)
    np.random.shuffle(train_idxs)
    
    val_idxs = np.arange(0, LEN_VAL_DATASET)
    np.random.shuffle(val_idxs)
    
    if LOAD_FLAG:
        model = load_model('logs/unet.hdf5', custom_objects={'focal_loss_fixed': uNet.focal_loss(gamma=2., alpha=.6)})
        print(model.summary())
    else:
        unet = uNet.uNet(160, 128, batch_size=BATCH_SIZE)
        model = unet.model
        print(model.summary())
    
    train_generator = batch.Batch(train_idxs, fr_train, batch_size=BATCH_SIZE, split='train')
    val_generator = batch.Batch(val_idxs, fr_val, batch_size=BATCH_SIZE, split='val')
    test_generator = batch.Batch(val_idxs, fr_val, batch_size=BATCH_SIZE, split='val')
    
    # Add a checkpoint for our model
    model_checkpoint = ModelCheckpoint('logs/unet.hdf5', monitor='loss', verbose=1, save_best_only=True)
    # save weights
    sw = model_tools.WeightsSaver(model)
    
    results = model.fit_generator(
                generator=train_generator.__start__(int(LEN_TRAIN_DATASET/BATCH_SIZE) - 1),
                validation_data=val_generator.__start__(int(LEN_VAL_DATASET/BATCH_SIZE) - 1),
                steps_per_epoch=int(LEN_TRAIN_DATASET/BATCH_SIZE),
                validation_steps=int(LEN_VAL_DATASET/BATCH_SIZE),
                epochs=20,
                callbacks=[model_checkpoint, sw])
    
if __name__=='__main__':
    import sys
    if len(sys.argv) > 1:
        assert sys.argv[1] == '--load'
        train(LOAD_FLAG=True)
    else:
        train()