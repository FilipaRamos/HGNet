import argparse
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

""" Custom imports """
import batch
import builder
from models import uNet
from models import model_tools

seed = 7
np.random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size during training [default: 64]')
parser.add_argument('--load', type=str, default=None, help='Load pre trained model')
parser.add_argument('--tresh', type=float, default=0.5, help='Tresh for accepting logits as belonging to the area of interest [default: 0.5')
parser.add_argument('--grid_x', type=int, default=160, help='Set a grid size on x')
parser.add_argument('--grid_y', type=int, default=128, help='Set a grid size on y')
parser.add_argument('--plot_epoch', type=int, default=10, help='The multiple value of epochs at which to plot loss relationship.')
parser.add_argument('--intensity', type=bool, default=False, help='Use intensity for training.') 
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
GRID_X = FLAGS.grid_x
GRID_Y = FLAGS.grid_y
TRESH = FLAGS.tresh
LOAD_FLAG = FLAGS.load
P_EPOCH = FLAGS.plot_epoch
INTENSITY = FLAGS.intensity

def train():    
    # Train and val pickle opened
    fr_train = builder.Frustum(GRID_X, GRID_Y, split='train', intensity=INTENSITY)
    fr_val = builder.Frustum(GRID_X, GRID_Y, split='val', intensity=INTENSITY)
    
    LEN_TRAIN_DATASET = len(fr_train)
    LEN_VAL_DATASET = len(fr_val)
    
    train_idxs = np.arange(0, LEN_TRAIN_DATASET)
    np.random.shuffle(train_idxs)
    
    val_idxs = np.arange(0, LEN_VAL_DATASET)
    np.random.shuffle(val_idxs)
    
    if INTENSITY:
        channels = 2
    else:
        channels = 1
    
    if LOAD_FLAG is not None:
        model = load_model('logs/' + LOAD_FLAG, custom_objects={'focal_loss_fixed': uNet.focal_loss(gamma=2., alpha=.6), 'precision_': uNet.precision(tresh=TRESH), 'recall_': uNet.recall(tresh=TRESH)})
        print(model.summary())
    else:
        unet = uNet.uNet(GRID_X, GRID_Y, channels, threshold=TRESH, batch_size=BATCH_SIZE, start_ch=int(GRID_X/2))
        model = unet.model
        print(model.summary())
    
    train_generator = batch.Batch(train_idxs, fr_train, batch_size=BATCH_SIZE, split='train')
    val_generator = batch.Batch(val_idxs, fr_val, batch_size=BATCH_SIZE, split='val')
    test_generator = batch.Batch(val_idxs, fr_val, batch_size=BATCH_SIZE, split='val')
    
    checkpoint_file = 'logs/unet' + str(GRID_X) +'-' + str(GRID_Y) + '.hdf5'
    
    # Add a checkpoint for our model
    model_checkpoint = ModelCheckpoint(checkpoint_file, monitor='loss', verbose=1, save_best_only=True)
    # save weights
    sw = model_tools.WeightsSaver(model, GRID_X, GRID_Y)
    # saving images from the validation set prediction at the end of each epoch
    saveImg = model_tools.SaveTrainEx(model, [0, 500, 1000], test_generator)
    # plot the losses values 
    pl = model_tools.PlotLosses(P_EPOCH)
    # save computation time by epoch
    time_callback = model_tools.TimeHistory()
    
    results = model.fit_generator(
                generator=train_generator.__start__(int(LEN_TRAIN_DATASET/BATCH_SIZE) - 1),
                validation_data=val_generator.__start__(int(LEN_VAL_DATASET/BATCH_SIZE) - 1),
                steps_per_epoch=int(LEN_TRAIN_DATASET/BATCH_SIZE),
                validation_steps=int(LEN_VAL_DATASET/BATCH_SIZE),
                epochs=100,
                callbacks=[model_checkpoint, sw, saveImg, pl, time_callback])
    
if __name__=='__main__':
    train()