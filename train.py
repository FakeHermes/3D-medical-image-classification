from mylib.dataloader.dataset import ClfDataset, get_balanced_loader, get_loader, get_x_test
from mylib.models import densesharp, metrics, losses
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv3D, MaxPooling3D
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import keras
import os
import _pickle as pickle
import numpy

def main(batch_sizes, crop_size, random_move, learning_rate,
         segmentation_task_ratio, weight_decay, save_folder, epochs):
    '''

    :param batch_sizes: the number of examples of each class in a single batch
    :param crop_size: the input size
    :param random_move: the random move in data augmentation
    :param learning_rate: learning rate of the optimizer
    :param segmentation_task_ratio: the weight of segmentation loss in total loss
    :param weight_decay: l2 weight decay
    :param save_folder: where to save the snapshots, tensorflow logs, etc.
    :param epochs: how many epochs to run
    :return:
    '''
    
    batch_size = sum(batch_sizes)
    
    train_dataset = ClfDataset(crop_size=crop_size, subset=[0, 1, 2], move=None,
                                  define_label=lambda l: [l[0] ,l[1]])

    val_dataset = ClfDataset(crop_size=crop_size, subset=[3], move=None,
                                define_label=lambda l: [l[0] ,l[1]])

    train_loader = get_balanced_loader(train_dataset, batch_sizes=batch_sizes)
    val_loader = get_loader(val_dataset, batch_size=batch_size)
    
    model = densesharp.get_compiled(output_size=2,
                                    optimizer=Adam(lr=learning_rate),
                                    loss='categorical_crossentropy',
                                    metrics=['accuracy'],
                                    weight_decay=weight_decay)
    
    checkpointer = ModelCheckpoint(filepath='tmp/%s/weights.{epoch:02d}.h5' % save_folder, verbose=1,
                                   period=1, save_weights_only=True)
    best_keeper = ModelCheckpoint(filepath='tmp/%s/best.h5' % save_folder, verbose=1, save_weights_only=True,
                                  monitor='val_clf_acc', save_best_only=True, period=1, mode='max')
    csv_logger = CSVLogger('tmp/%s/training.csv' % save_folder)
    tensorboard = TensorBoard(log_dir='tmp/%s/logs/' % save_folder)
    early_stopping = EarlyStopping(monitor='val_clf_acc', min_delta=0, mode='max',
                                   patience=30, verbose=1)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.334, patience=10,
                                   verbose=1, mode='min', epsilon=1.e-5, cooldown=2, min_lr=0)
    

    #model.fit_generator(generator=train_loader, steps_per_epoch=len(train_dataset), max_queue_size=500, workers=1,
     #                   validation_data=val_loader, epochs=epochs, validation_steps=len(val_dataset))
                        
    model.fit_generator(generator=train_loader, steps_per_epoch=len(train_dataset), max_queue_size=500, workers=1,
                        validation_data=val_loader, epochs=epochs, validation_steps=len(val_dataset),
                        callbacks=[checkpointer, early_stopping, best_keeper, lr_reducer, csv_logger])
    #save model
    with open('cnns1225.pickle','wb') as file:
        pickle.dump(model,file)

    test=get_x_test()
    print(np.shape(test))
 
    y=model.predict(test)
    for i in range(117):
        print(y[i,1])

if __name__ == '__main__':
    main(batch_sizes=[12,10],
         crop_size=[32, 32, 32],
         random_move=3,
         learning_rate=1.e-4,
         segmentation_task_ratio=0.2,
         weight_decay=0,
         save_folder='test',
         epochs=10)
