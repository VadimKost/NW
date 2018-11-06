import numpy as np
from keras.datasets import mnist
from keras.models import Sequential,model_from_json
from keras.layers import Dense
from keras.utils import np_utils
from keras.preprocessing import image

file= open('model.json','r')
loaded= file.read()
file.close()

model= model_from_json(loaded)
model.load_weights("model.h5")
# Компилируем модель
model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train=X_train.astype('float32')


img_path = '1.jpg'
img = image.load_img(img_path, target_size=(28,28),grayscale=True)

x = image.img_to_array(img)
x=x.reshape(1,784)
x = 255 - x


print(x)





find = model.predict(x)
find=np_utils.to_categorical(find)
print(find)
fl=list(find)
print('=======Відповідь=========')
for (index,i) in enumerate(fl[0]):
    if i[1]==1:
        result=index
        print(result)









