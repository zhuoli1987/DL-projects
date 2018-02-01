from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50
from extract_bottleneck_features import *
from keras.models import load_model
from train_model import *
from glob import glob
import cv2 

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# define Resnet50 model for dog breed
try:
    Resnet50_model = load_model('model_weights/weights.best.Resnet50.hdf5')
except OSError:
    Resnet50_model_train()
    Resnet50_model = load_model('model_weights/weights.best.Resnet50.hdf5')

# load pretrained Resnet50 model for dog claasification
Resnet50_model_2 = ResNet50(weights='imagenet')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]


### methods
def image_to_tensor(img):
    if img.mode != 'RGB':
        img = img.convert('RGB') 
    # resize the image
    img = img.resize((224,224))
	# convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
	# convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    output = np.expand_dims(x, axis=0)
    # set the tensor to be writable
    output.setflags(write=1)
    return output

def Resnet50_predict_breed(img):
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(image_to_tensor(img))
    # obtain predicted vector
    predicted_vector = Resnet50_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]

def ResNet50_predict_labels(img):
    # returns prediction vector for image located at img_path
    img = preprocess_input(image_to_tensor(img))
    return np.argmax(Resnet50_model_2.predict(img))

def face_detector(img):
    x = np.asarray(img)
    gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def dog_detector(img):
    prediction = ResNet50_predict_labels(img)
    return ((prediction <= 268) & (prediction >= 151)) 




