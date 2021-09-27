
from flask import Flask , render_template, request
import numpy as np
import pandas as pdpip
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.transform import resize
import cv2


potatoes_classes=['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

pepper_calsses=['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy']

tommatoes_classes = ['Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']


 
potatoes_model=tf.keras.models.load_model('../models/1')
tommatoes_model=tf.keras.models.load_model('../tomato _model/1')
pepper_model=tf.keras.models.load_model('../pepper_model/1')


app = Flask(__name__)



@app.route('/')
def index():
	return render_template("home.html")

@app.route("/tom_prediction", methods=["POST"])

def tom_prediction():
	
	model=tommatoes_model
	classes=tommatoes_classes
	

	img = request.files['img']
	

	img.save("img.jpg")

	image = cv2.imread("img.jpg")

	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	image = cv2.resize(image, (256,256))

	image = np.reshape(image, (1,256,256,3))

	
	pred = model.predict(image)
	conf = round(100 * (np.max(pred[0])))
	pred = np.argmax(pred)

	pred = classes[pred]
    

	return render_template("prediction.html", data=pred,conf=float(conf))
@app.route("/pot_prediction", methods=["POST"])

def pot_prediction():
	
	model=potatoes_model
	classes=potatoes_classes
	

	img = request.files['img']
	

	img.save("img.jpg")

	image = cv2.imread("img.jpg")

	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	image = cv2.resize(image, (256,256))

	image = np.reshape(image, (1,256,256,3))

	
	pred = model.predict(image)
	conf = round(100 * (np.max(pred[0])))
	pred = np.argmax(pred)

	pred = classes[pred]
    

	return render_template("prediction.html", data=pred,conf=float(conf))
@app.route("/pep_prediction", methods=["POST"])

def pep_prediction():
	
	model=pepper_model
	classes=pepper_calsses
	

	img = request.files['img']
	

	img.save("img.jpg")

	image = cv2.imread("img.jpg")

	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	image = cv2.resize(image, (256,256))

	image = np.reshape(image, (1,256,256,3))

	
	pred = model.predict(image)
	conf = round(100 * (np.max(pred[0])))
	pred = np.argmax(pred)

	pred = classes[pred]
    

	return render_template("prediction.html", data=pred,conf=float(conf))



if __name__=="__main__":
    app.run(debug=True) 