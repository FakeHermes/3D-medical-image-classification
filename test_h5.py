from keras.models import load_model
from mylib.dataloader.dataset import ClfDataset, get_balanced_loader, get_loader, get_x_test
from mylib.models import densesharp, metrics, losses
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv3D, MaxPooling3D
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import keras
import os
import _pickle as pickle
import numpy as np
def main():
	model = densesharp.get_compiled(output_size=2,
									optimizer=Adam(lr=1.e-4),
									loss='categorical_crossentropy',
									metrics=['accuracy'],
									weight_decay=0)
	model.load_weights('result/p4.h5')
	with open('result/p4.pickle','wb') as file:
		pickle.dump(model,file)
	test=get_x_test()
	print(np.shape(test))
	
	y=model.predict(test)
	for i in range(117):
		print(y[i,1])
		
if __name__ == '__main__':
    main()
	