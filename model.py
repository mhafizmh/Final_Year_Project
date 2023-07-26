import os
import cv2
import numpy
import imageio
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras import backend as K

K.set_image_data_format('channels_first')       

image_rows, image_columns, image_depth = 64, 64,240

training_list = []
angrypath = 'E:/FYP Dataset/New Combined/anger'
happypath = 'E:/FYP Dataset/New Combined/happiness'
disgustpath = 'E:/FYP Dataset/New Combined/disgust'
sadpath = 'E:/FYP Dataset/New Combined/sadness'
fearpath = 'E:/FYP Dataset/New Combined/fear'
surprisepath = 'E:/FYP Dataset/New Combined/surprise'
neutralpath = 'E:/FYP Dataset/New Combined/neutral'

directorylisting = os.listdir(angrypath)
countangry = 0
for video in directorylisting:
	frames = []
	videopath = angrypath + '\\' + video
	loadedvideo = imageio.get_reader(videopath, 'ffmpeg')
	framerange = [x + 72 for x in range(96)]
	for frame in framerange:
		try:
			image = loadedvideo.get_data(frame)
		except:
			# Add padding instead of raising an exception
			image = numpy.zeros((image_rows, image_columns, 3), dtype=numpy.uint8)
		imageresize = cv2.resize(image, (image_rows, image_columns), interpolation = cv2.INTER_AREA)
		grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
		frames.append(grayimage)
	frames = numpy.asarray(frames)
	videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
	training_list.append(videoarray)
	countangry += 1

directorylisting = os.listdir(happypath)
counthappy = 0
for video in directorylisting:
	frames = []
	videopath = happypath + video
	loadedvideo = imageio.get_reader(videopath, 'ffmpeg')
	framerange = [x + 72 for x in range(96)]
	for frame in framerange:
		try:
			image = loadedvideo.get_data(frame)
		except:
			# Add padding instead of raising an exception
			image = numpy.zeros((image_rows, image_columns, 3), dtype=numpy.uint8)
		imageresize = cv2.resize(image, (image_rows, image_columns), interpolation = cv2.INTER_AREA)
		grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
		frames.append(grayimage)		
	frames = numpy.asarray(frames)
	videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
	training_list.append(videoarray)
	counthappy += 1
 
directorylisting = os.listdir(disgustpath)
countdisgust = 0
for video in directorylisting:
	frames = []
	videopath = disgustpath + video
	loadedvideo = imageio.get_reader(videopath, 'ffmpeg')
	framerange = [x + 72 for x in range(96)]
	for frame in framerange:
		try:
			image = loadedvideo.get_data(frame)
		except:
			# Add padding instead of raising an exception
			image = numpy.zeros((image_rows, image_columns, 3), dtype=numpy.uint8)
		imageresize = cv2.resize(image, (image_rows, image_columns), interpolation = cv2.INTER_AREA)
		grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
		frames.append(grayimage)
	frames = numpy.asarray(frames)
	videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
	training_list.append(videoarray)
	countdisgust += 1
	

directorylisting = os.listdir(sadpath)
countsad =0
for video in directorylisting:
	frames = []
	videopath = sadpath + video
	loadedvideo = imageio.get_reader(videopath, 'ffmpeg')
	framerange = [x + 72 for x in range(96)]
	for frame in framerange:
		try:
			image = loadedvideo.get_data(frame)
		except:
			# Add padding instead of raising an exception
			image = numpy.zeros((image_rows, image_columns, 3), dtype=numpy.uint8)
		imageresize = cv2.resize(image, (image_rows, image_columns), interpolation = cv2.INTER_AREA)
		grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
		frames.append(grayimage)
	frames = numpy.asarray(frames)
	videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
	training_list.append(videoarray)
	countsad += 1
	
directorylisting = os.listdir(fearpath)
countfear =  0
for video in directorylisting:
	frames = []
	videopath = fearpath + video
	loadedvideo = imageio.get_reader(videopath, 'ffmpeg')
	framerange = [x + 72 for x in range(96)]
	for frame in framerange:
		try:
			image = loadedvideo.get_data(frame)
		except:
			# Add padding instead of raising an exception
			image = numpy.zeros((image_rows, image_columns, 3), dtype=numpy.uint8)
		imageresize = cv2.resize(image, (image_rows, image_columns), interpolation = cv2.INTER_AREA)
		grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
		frames.append(grayimage)
	frames = numpy.asarray(frames)
	videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
	training_list.append(videoarray)
	countfear += 1
 
directorylisting = os.listdir(surprisepath)
countsurprise =  0
for video in directorylisting:
	frames = []
	videopath = surprisepath + video
	loadedvideo = imageio.get_reader(videopath, 'ffmpeg')
	framerange = [x + 72 for x in range(96)]
	for frame in framerange:
		try:
			image = loadedvideo.get_data(frame)
		except:
			# Add padding instead of raising an exception
			image = numpy.zeros((image_rows, image_columns, 3), dtype=numpy.uint8)
		imageresize = cv2.resize(image, (image_rows, image_columns), interpolation = cv2.INTER_AREA)
		grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
		frames.append(grayimage)
	frames = numpy.asarray(frames)
	videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
	training_list.append(videoarray)
	countsurprise += 1
	
 
training_list = numpy.asarray(training_list, dtype = object)
trainingsamples = len(training_list)

traininglabels = numpy.zeros((trainingsamples, ), dtype = int)


traininglabels[0:countangry] = 0
traininglabels[countangry:countangry + counthappy] = 1
traininglabels[countangry + counthappy:countangry + counthappy + countdisgust] = 2
traininglabels[countangry + counthappy + countdisgust:countangry + counthappy + countdisgust + countsad] = 3
traininglabels[countangry + counthappy + countdisgust + countsad: countangry + counthappy + countdisgust + countsad + countfear] = 4
traininglabels[countangry + counthappy + countdisgust + countsad: countangry + counthappy + countdisgust + countsad + countfear + countsurprise] = 5
traininglabels = np_utils.to_categorical(traininglabels, 6)

training_data = [training_list, traininglabels]
(trainingframes, traininglabels) = (training_data[0], training_data[1])
training_set = numpy.zeros((trainingsamples, 1, image_rows, image_columns, image_depth))
for h in range  (trainingsamples):
	training_set[h][0][:][:][:] = trainingframes[h,:,:,:]

training_set = training_set.astype('float32')
training_set -= numpy.mean(training_set)
training_set /= numpy.max(training_set)

# Save training images and labels in a numpy array
numpy.save('numpy_training_dataset/train_images.npy', training_set)
numpy.save('numpy_training_dataset/train_labels.npy', traininglabels)

# Load training images and labels that are stored in numpy array
training_set = numpy.load('numpy_training_dataset/train_images.npy')
traininglabels =numpy.load('numpy_training_dataset/train_labels.npy')


# 3D CNN Model
model = Sequential()
model.add(Convolution3D(32, (3, 3, 15), strides=(1, 1, 1), padding='same', data_format='channels_last', input_shape=(1, image_rows, image_columns, image_depth), activation='relu'))
model.add(MaxPooling3D(pool_size=(3, 3, 3)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, kernel_initializer = 'random_normal', bias_initializer='zeros', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, kernel_initializer = 'random_normal', bias_initializer='zeros'))
model.add(Activation('softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'SGD', metrics = ['accuracy'])
	

filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]



# Spliting the dataset into training and validation sets
train_images, validation_images, train_labels, validation_labels =  train_test_split(training_set, traininglabels, test_size=0.2, random_state=4)

# Save validation set in a numpy array
numpy.save('numpy_validation_dataset/val_images.npy', validation_images)
numpy.save('numpy_validation_dataset/val_labels.npy', validation_labels)

# Load validation set from numpy array
validation_images = numpy.load('numpy_validation_dataset/val_images.npy')
validation_labels = numpy.load('numpy_validation_dataset/val_labels.npy')


# Training the model
hist = model.fit(train_images, train_labels, validation_data = (validation_images, validation_labels), callbacks=callbacks_list, batch_size = 16, epochs = 100, shuffle=True)

# Finding Confusion Matrix of model

predictions = model.predict(validation_images)
predictions_labels = numpy.argmax(predictions, axis=1)
validation_labels = numpy.argmax(validation_labels, axis=1)
cfm = confusion_matrix(validation_labels, predictions_labels)
print (cfm)

