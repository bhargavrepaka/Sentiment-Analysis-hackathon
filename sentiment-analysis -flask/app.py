from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from keras.models import load_model
import numpy as np
import cv2
import keras
import tensorflow as tf
# Utils
import joblib 
pipe_lr = joblib.load(open("emotion_classifier_pipe_lr.pkl","rb"))


# Fxn
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    label_dict={'happy':'Positive','sad':'Negative','fear':'Negative','anger': "Negative",'surprise':'Positive','neutral':'Neutral','disgust':'Negative','shame':"Neutal"  }
    return label_dict[results[0]]

def get_prediction_proba(docx):
	results = pipe_lr.predict_proba([docx])
	return results




model = tf.keras.models.load_model('facialemotionmodel.h5')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# model = load_model('model2.h5')
# model.save('model2_updated.h5', save_format='h5')
# custom_objects = {'InputLayer': keras.layers.InputLayer}
# model = load_model('model2_updated.h5', custom_objects=custom_objects)

@app.route ('/upload_text',methods=['POST'])
def upload_text():
    raw_text=request.data
    print(raw_text)
    final_prediction=predict_emotions(raw_text)
    return jsonify({'success': True, 'emotion': final_prediction})


@app.route('/upload_image', methods=['POST', 'OPTIONS'])
def upload_image():
    if request.method == 'OPTIONS':
        # Respond to pre-flight requests
        response = app.make_default_options_response()
        response.headers['Access-Control-Allow-Methods'] = 'POST'
        return response

    try:
        # Check if the 'image' key is in the request files
        if 'image' in request.files:
            image_file = request.files['image']

            # Generate a unique filename
            filename = 'captured_image.jpg'  # You can create a more elaborate filename if needed

            # Save the image to the uploads folder
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(filepath)

            print('Saved Filepath:', filepath)  # Log the saved filepath


            image = cv2.imread('uploads/captured_image.jpg', 0)

            img1 = cv2.imread('uploads/captured_image.jpg')
            gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
            faces = cascade.detectMultiScale(gray, 1.1, 3)

            for x,y,w,h in faces:
               cv2.rectangle(img1, (x,y), (x+w, y+h), (0,255,0), 2)

               cropped = img1[y:y+h, x:x+w]

            cv2.imwrite('uploads/after.jpg', img1)

            try:
               cv2.imwrite('uploads/cropped.jpg', cropped)

            except:
               pass

    #####################################

            try:
               image = cv2.imread('uploads/cropped.jpg', 0)
            except:
               image = cv2.imread('uploads/file.jpg', 0)
            # image=gray
            image = cv2.resize(image, (48,48))

            image = image/255.0

            image = np.reshape(image, (1,48,48,1))

            model = tf.keras.models.load_model('model2.h5')

            prediction = model.predict(image)

            label_map =   ['Anger','Neutral' , 'Fear', 'Happy', 'Sad', 'Surprise']

            prediction = np.argmax(prediction)

            final_prediction = label_map[prediction]
            label_dict = {'Anger':'Negative', 'Neutral':'Neutral', 'Fear':'Negative', 'Happy':'Positive', 'Sad':'Negative', 'Surprise':'Positive'}
            print(label_dict[final_prediction])

            return jsonify({'success': True, 'emotion':label_dict[final_prediction]})
        else:
            return jsonify({'success': False, 'error': 'No file provided'})

    except Exception as e:
        print('Error:', str(e))  # Log any errors
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Create the 'uploads' folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    app.run(debug=True)
