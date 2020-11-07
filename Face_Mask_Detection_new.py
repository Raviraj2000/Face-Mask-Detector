import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from keras.preprocessing.image import ImageDataGenerator


train_dir = "Dataset/train"

train_datagen = ImageDataGenerator(rescale = 1.0/255,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range=0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)


train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = 25, 
                                                    target_size=(150, 150),
                                                    )

test_dir = "Dataset/test"

test_datagen = ImageDataGenerator(rescale = 1.0/255)

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  batch_size = 25,
                                                  target_size = (150, 150),
                                                  )


from keras.models import Sequential
from keras.layers import Dense, Conv2D, AveragePooling2D, Dropout, Flatten


model = Sequential()

model.add(Conv2D(128, kernel_size = (3,3), activation = 'relu', input_shape = (150,150,3)))
model.add(AveragePooling2D(pool_size = (2,2)))

model.add(Conv2D(256, kernel_size = (3,3), activation = 'relu'))
model.add(AveragePooling2D(pool_size = (2,2)))

model.add(Conv2D(256, kernel_size = (3,3), activation = 'relu'))
model.add(AveragePooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation = 'softmax'))


model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.summary()

history = model.fit(train_generator, 
                    epochs = 10,
                    validation_data = test_generator
                    )

#model.save_weights('Face_Mask_model_black.h5')

#model.load_weights('Face_Mask_model.h5')


import cv2

cv2.ocl.setUseOpenCL(False)
results={0:'without mask',1:'mask'}
GR_dict={0:(0,0,255),1:(0,255,0)}
cap = cv2.VideoCapture(0)

while True:
    ret, im = cap.read()
    if not ret:
        break
    
    bounding_box = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    num_faces = bounding_box.detectMultiScale(im) 
    
    for (x, y, w, h) in num_faces:
        face_img = im[y:y+h, x:x+w]
        rerect_sized = cv2.resize(face_img, (150,150))
        normalized = rerect_sized/255.0
        reshaped = np.reshape(normalized, (1,150,150,3))
        reshaped = np.vstack([reshaped])
        result = model.predict(reshaped)
        
        label = np.argmax(result)
        
        cv2.rectangle(im, (x, y), (x+w, y+h), GR_dict[label], 2)
        cv2.rectangle(im, (x, y-40), (x+w, y), GR_dict[label], -1)
        cv2.putText(im, results[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
    cv2.imshow('Video', cv2.resize(im,(1200,860),interpolation = cv2.INTER_CUBIC))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
    

        
        
        
        
        
        
        
        
        
    
    

























