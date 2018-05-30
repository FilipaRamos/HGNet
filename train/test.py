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

# for now just load and predict for one height grid
# TODO save all frustum height grids as images?
def test(model='/logs/unet.hdf5', weights='/logs/weights00.h5', test_obj=None):
    
    # load the model
    model = load_model(model, custom_objects={'focal_loss_fixed': uNet.focal_loss(gamma=2., alpha=.6)})
    print(model.summary())
    
    model.predict(test_obj)