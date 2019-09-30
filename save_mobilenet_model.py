import numpy as np
from keras.models import load_model
from keras.applications.mobilenet import MobileNet
from keras.applications.resnet50 import decode_predictions

# Mobile Net
model = MobileNet(weights="imagenet", include_top=True)

# モデルの保存、モデルをロードした後予測しかしないため、include_optimizer=Falseとする
model.save('test.h5', include_optimizer=False)