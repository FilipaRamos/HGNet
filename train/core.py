import numpy as np

import batch
import builder

from keras.callbacks import ModelCheckpoint
from keras.models import load_model

""" Custom imports """
from visualization import tools

from models import uNet
from models import model_tools
from region import objArea

seed = 7
np.random.seed(seed)
BATCH_SIZE=1

def test(index=0):
    # Train and val pickle opened
    fr_train = builder.Frustum(split='train')
    fr_val = builder.Frustum(split='val')
    
    LEN_TRAIN_DATASET = len(fr_train)
    LEN_VAL_DATASET = len(fr_val)
    
    train_idxs = np.arange(0, LEN_TRAIN_DATASET)
    np.random.shuffle(train_idxs)
    
    val_idxs = np.arange(0, LEN_VAL_DATASET)
    np.random.shuffle(val_idxs)
    
    train_generator = batch.Batch(train_idxs, fr_train, batch_size=BATCH_SIZE, split='train', offsets=True)
    val_generator = batch.Batch(val_idxs, fr_val, batch_size=BATCH_SIZE, split='val', offsets=True)
    
    # get a sample grid
    X_, y_, offsets_ = val_generator.__samples__(index)
    X = np.squeeze(X_)
    y = np.squeeze(y_)
    
    model = load_model('logs/unet.hdf5', custom_objects={'focal_loss_fixed': uNet.focal_loss(gamma=2., alpha=.6), 'f1': uNet.f1})
    print(model.summary())
    
    logits_ = model.predict(X_)
    logits = np.squeeze(logits_)
    tools.plot_grid_colormap(X, y, pred=logits)
    
    # object area to extract connected components
    objA = objArea.objArea(X, logits, offsets_)
    objA.__filter_seg__()
    objA.__cc__()
    
    tools.plot_grid_colormap(X, y, pred=objA.components_grid)
    
    
if __name__=='__main__':
    import sys
    if len(sys.argv) > 1:
        test(index=int(sys.argv[1]))
    else:
        test()