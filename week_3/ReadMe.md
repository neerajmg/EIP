
### Base network

Accuracy on test data is: 82.96


### Modified Model

python
dropout = 0.1
model = Sequential()
model.add(InputLayer(train_features.shape[1:]))

def conv(out_size, **kwargs):
    model.add(SeparableConv2D(out_size, 3, padding='same', **kwargs))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

def maxp():
    model.add(MaxPooling2D(2))
    model.add(Dropout(dropout))

conv(64) # output size = 32 x 32 x 64 | receptive field = 3

conv(64) # output size = 32 x 32 x 64 | receptive field = 5

maxp() # output size = 16 x 16 x 64 | receptive field = 6

conv(96) # output size = 16 x 16 x 96 | receptive field = 10

conv(96) # output size = 16 x 16 x 96 | receptive field = 14

maxp() # output size = 8 x 8 x 96 | receptive field = 16

conv(108) # output size = 8 x 8 x 108 | receptive field = 24

conv(108) # output size = 8 x 8 x 108 | receptive field = 32

maxp() # output size = 4 x 4 x 108 | receptive field = 36

conv(128) # output size = 4 x 4 x 128 | receptive field = 52

conv(128) # output size = 4 x 4 x 128 | receptive field = 68

maxp() # output size = 2 x 2 x 128 | receptive field = 76

conv(98) # output size = 2 x 2 x 98 | receptive field = 108

maxp() # output size = 1 x 1 x 98 | receptive field = 124

model.add(SeparableConv2D(num_classes, 3, padding='same', activation='softmax')) # output size = 1 x 1 x 10 | receptive field = 188
model.add(Flatten())

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)
return model


### Modified model results

Epoch 1/50
390/390 [==============================] - 76s 195ms/step - loss: 1.5864 - acc: 0.4187 - val_loss: 2.3093 - val_acc: 0.3563
Epoch 2/50
390/390 [==============================] - 38s 98ms/step - loss: 1.1384 - acc: 0.5927 - val_loss: 1.3032 - val_acc: 0.5774
Epoch 3/50
390/390 [==============================] - 38s 98ms/step - loss: 0.9945 - acc: 0.6466 - val_loss: 0.9522 - val_acc: 0.6667
Epoch 4/50
390/390 [==============================] - 38s 97ms/step - loss: 0.8996 - acc: 0.6794 - val_loss: 1.1868 - val_acc: 0.5915
Epoch 5/50
390/390 [==============================] - 38s 97ms/step - loss: 0.8352 - acc: 0.7068 - val_loss: 1.0512 - val_acc: 0.6527
Epoch 6/50
390/390 [==============================] - 38s 98ms/step - loss: 0.7929 - acc: 0.7214 - val_loss: 0.9189 - val_acc: 0.6820
Epoch 7/50
390/390 [==============================] - 38s 97ms/step - loss: 0.7465 - acc: 0.7386 - val_loss: 0.9655 - val_acc: 0.6813
Epoch 8/50
390/390 [==============================] - 38s 97ms/step - loss: 0.7106 - acc: 0.7507 - val_loss: 1.1055 - val_acc: 0.6424
Epoch 9/50
390/390 [==============================] - 38s 97ms/step - loss: 0.6917 - acc: 0.7581 - val_loss: 0.7703 - val_acc: 0.7295
Epoch 10/50
390/390 [==============================] - 38s 97ms/step - loss: 0.6646 - acc: 0.7673 - val_loss: 0.7073 - val_acc: 0.7611
Epoch 11/50
390/390 [==============================] - 37s 96ms/step - loss: 0.6455 - acc: 0.7737 - val_loss: 0.7181 - val_acc: 0.7554
Epoch 12/50
390/390 [==============================] - 37s 96ms/step - loss: 0.6287 - acc: 0.7824 - val_loss: 0.7655 - val_acc: 0.7391
Epoch 13/50
390/390 [==============================] - 37s 96ms/step - loss: 0.6081 - acc: 0.7885 - val_loss: 0.8606 - val_acc: 0.7145
Epoch 14/50
390/390 [==============================] - 37s 96ms/step - loss: 0.5964 - acc: 0.7949 - val_loss: 0.6653 - val_acc: 0.7762
Epoch 15/50
390/390 [==============================] - 38s 96ms/step - loss: 0.5841 - acc: 0.7943 - val_loss: 0.7786 - val_acc: 0.7389
Epoch 16/50
390/390 [==============================] - 38s 97ms/step - loss: 0.5688 - acc: 0.8049 - val_loss: 0.7030 - val_acc: 0.7672
Epoch 17/50
390/390 [==============================] - 38s 97ms/step - loss: 0.5603 - acc: 0.8064 - val_loss: 0.7352 - val_acc: 0.7563
Epoch 18/50
390/390 [==============================] - 37s 96ms/step - loss: 0.5512 - acc: 0.8084 - val_loss: 0.7249 - val_acc: 0.7594
Epoch 19/50
390/390 [==============================] - 37s 96ms/step - loss: 0.5376 - acc: 0.8129 - val_loss: 0.6254 - val_acc: 0.7884
Epoch 20/50
390/390 [==============================] - 38s 96ms/step - loss: 0.5325 - acc: 0.8150 - val_loss: 0.6029 - val_acc: 0.7943
Epoch 21/50
390/390 [==============================] - 37s 96ms/step - loss: 0.5189 - acc: 0.8193 - val_loss: 0.6010 - val_acc: 0.8022
Epoch 22/50
390/390 [==============================] - 38s 96ms/step - loss: 0.5180 - acc: 0.8192 - val_loss: 0.5499 - val_acc: 0.8171
Epoch 23/50
390/390 [==============================] - 37s 96ms/step - loss: 0.5060 - acc: 0.8243 - val_loss: 0.5543 - val_acc: 0.8149
Epoch 24/50
390/390 [==============================] - 38s 97ms/step - loss: 0.4971 - acc: 0.8271 - val_loss: 0.7547 - val_acc: 0.7627
Epoch 25/50
390/390 [==============================] - 38s 97ms/step - loss: 0.4890 - acc: 0.8299 - val_loss: 0.6458 - val_acc: 0.7884
Epoch 26/50
390/390 [==============================] - 37s 95ms/step - loss: 0.4875 - acc: 0.8305 - val_loss: 0.5195 - val_acc: 0.8251
Epoch 27/50
390/390 [==============================] - 37s 96ms/step - loss: 0.4805 - acc: 0.8317 - val_loss: 0.5745 - val_acc: 0.8097
Epoch 28/50
390/390 [==============================] - 38s 96ms/step - loss: 0.4781 - acc: 0.8343 - val_loss: 0.6241 - val_acc: 0.7888
Epoch 29/50
390/390 [==============================] - 38s 96ms/step - loss: 0.4680 - acc: 0.8379 - val_loss: 0.5615 - val_acc: 0.8102
Epoch 30/50
390/390 [==============================] - 37s 96ms/step - loss: 0.4634 - acc: 0.8394 - val_loss: 0.5880 - val_acc: 0.8021
Epoch 31/50
390/390 [==============================] - 38s 96ms/step - loss: 0.4584 - acc: 0.8412 - val_loss: 0.5514 - val_acc: 0.8166
Epoch 32/50
390/390 [==============================] - 37s 96ms/step - loss: 0.4538 - acc: 0.8426 - val_loss: 0.7128 - val_acc: 0.7681
Epoch 33/50
390/390 [==============================] - 37s 95ms/step - loss: 0.4511 - acc: 0.8435 - val_loss: 0.6359 - val_acc: 0.7902
Epoch 34/50
390/390 [==============================] - 37s 94ms/step - loss: 0.4427 - acc: 0.8451 - val_loss: 0.5333 - val_acc: 0.8218
Epoch 35/50
390/390 [==============================] - 37s 96ms/step - loss: 0.4385 - acc: 0.8478 - val_loss: 0.4950 - val_acc: 0.8342
Epoch 36/50
390/390 [==============================] - 37s 95ms/step - loss: 0.4345 - acc: 0.8489 - val_loss: 0.4987 - val_acc: 0.8359
Epoch 37/50
390/390 [==============================] - 37s 95ms/step - loss: 0.4309 - acc: 0.8502 - val_loss: 0.5755 - val_acc: 0.8069
Epoch 38/50
390/390 [==============================] - 37s 95ms/step - loss: 0.4294 - acc: 0.8495 - val_loss: 0.4996 - val_acc: 0.8318
Epoch 39/50
390/390 [==============================] - 37s 95ms/step - loss: 0.4283 - acc: 0.8508 - val_loss: 0.5253 - val_acc: 0.8251
Epoch 40/50
390/390 [==============================] - 37s 95ms/step - loss: 0.4244 - acc: 0.8533 - val_loss: 0.5447 - val_acc: 0.8195
Epoch 41/50
390/390 [==============================] - 37s 94ms/step - loss: 0.4178 - acc: 0.8532 - val_loss: 0.5615 - val_acc: 0.8143
Epoch 42/50
390/390 [==============================] - 39s 101ms/step - loss: 0.4077 - acc: 0.8562 - val_loss: 0.5681 - val_acc: 0.8157
Epoch 43/50
390/390 [==============================] - 37s 96ms/step - loss: 0.4144 - acc: 0.8561 - val_loss: 0.4992 - val_acc: 0.8352
Epoch 44/50
390/390 [==============================] - 38s 96ms/step - loss: 0.4109 - acc: 0.8569 - val_loss: 0.4880 - val_acc: 0.8347
Epoch 45/50
390/390 [==============================] - 38s 97ms/step - loss: 0.4016 - acc: 0.8606 - val_loss: 0.5416 - val_acc: 0.8241
Epoch 46/50
390/390 [==============================] - 37s 96ms/step - loss: 0.4016 - acc: 0.8600 - val_loss: 0.4673 - val_acc: 0.8432
Epoch 47/50
390/390 [==============================] - 38s 96ms/step - loss: 0.4005 - acc: 0.8606 - val_loss: 0.7124 - val_acc: 0.7760
Epoch 48/50
390/390 [==============================] - 38s 97ms/step - loss: 0.3975 - acc: 0.8624 - val_loss: 0.4836 - val_acc: 0.8390
Epoch 49/50
390/390 [==============================] - 38s 97ms/step - loss: 0.3901 - acc: 0.8637 - val_loss: 0.5529 - val_acc: 0.8188
Epoch 50/50
390/390 [==============================] - 38s 97ms/step - loss: 0.3891 - acc: 0.8639 - val_loss: 0.4691 - val_acc: 0.8443
Model took 1917.33 seconds to train

Accuracy on test data is: 84.43
