from __future__ import division
import numpy as np
from scipy.misc import imresize

def postprocess_predictions(pred, shape_r, shape_c):
    pred = imresize(pred, (shape_r, shape_c))
    pred = pred / np.max(pred) * 255
    return pred