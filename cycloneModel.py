import os
from skimage.io import imread
from skimage.transform import resize
from PIL import Image, ImageOps, ImageFile
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

ImageFile.LOAD_TRUNCATED_IMAGES = True

#PROCESS DATA AND CREATE MASK IMAGES
X = []
Y = []

locationFileNames = []
imageFileNames = []
size = 256

directoryCentLoc= r"data\center_locations"
directoryImages = r"data\images"

#create list with location center file names 
for file in os.listdir(directoryCentLoc):
    locationFileNames.append(file)
#create list with EPIC image file names 
for file in os.listdir(directoryImages):
    imageFileNames.append(file)

# loops through each EPIC image, creates corresponding mask image, adds to X and Y lists
for file in imageFileNames:
    imageName = file[:22] 
    cyclones = []
    centerCoords = []
    
    # checks if corresponding center location files exists and adds to list
    for txtFile in locationFileNames:
        if txtFile.startswith(imageName):
            cyclones.append(txtFile)    
    if len(cyclones) == 0:
        continue
    
    # reads center location files and adds string contaning cyclone center to list
    for cyclone in cyclones:
        cyclonePath = os.path.join(directoryCentLoc, cyclone)
        cycloneFile = open(cyclonePath, 'r')
        coord = cycloneFile.readlines()[1]
        cycloneFile.close()
        centerCoords.append(coord)
        
    #creates default mask image set to all black  
    mask = Image.new(mode = 'RGB', size = (2048,2048),color = (0,0,0))
    pixels = mask.load()
    
    #add cyclone bounding boxes to default mask image
    for center in centerCoords:
        #processes string containing cyclone center and extracts center
        coords = center.split(", ", 2)[:2]
        x_center = int(coords[0])
        y_center = int(coords[1])

        #creating boundary box around the extracted center and sets to all white
        i = 0
        while i < 101:
            x = i-50+x_center
            if (x<0 or x>2048):
                continue
            j = 0
            i = i+1
            while j < 101:
                y = j-50+y_center
                if (y<0 or y>2048):
                    continue
                pixels[x,y] = (255,255,255)
                j = j+1 
                
    # crops and resizes mask image and add to Y
    mask = mask.crop((384,1024,1664,1792))
    mask = mask.resize((size,size))             
    mask = ImageOps.grayscale(mask)
    mask = np.array(mask) # 256 by 256, from 0 to 255
    Y.append(mask) 
    
    #extract rgb array out of EPIC image, crops, resizes and add to X
    imagePath = os.path.join(directoryImages, file)
    img = imread(imagePath)[:,:,:3]
    img = img[384:1664, 1024:1792, :]
    img = resize(img, (size,size), mode='constant', preserve_range=True)
    X.append(img)
    
#split into testing and training sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state= 69)


#format data to be inputted into model
y_train = np.asarray(y_train)/255
x_train = np.asarray(x_train)/255
x_test = np.asarray(x_test)/255
y_test = np.asarray(y_test)/255





#UNET MODEL


inputs = tf.keras.layers.Input((size, size, 3))


#Contraction path
c1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)

c3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)

c4 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)

c5 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#Expansive path
u6 = tf.keras.layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(16, (2,2), strides=(2,2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

outputs = tf.keras.layers.Conv2D(1, (1,1), activation='sigmoid')(c9)


#training model 
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',tf.keras.metrics.BinaryIoU(name = 'IoU')])
model.summary()
results = model.fit(x_train, y_train, batch_size=16, epochs=3)

#testing model
scores = model.evaluate(x_test, y_test)






