from flask import Flask, render_template, request
from keras.preprocessing.image import img_to_array
from keras.applications.mobilenet import preprocess_input
import numpy as np
import tensorflow as tf
import base64 #encoding image through http request
from PIL import Image #open image
from io import BytesIO #hold image without saving in local disk

app = Flask(__name__)
model = None
class_labels = None
predicted_letters = []

def load_saved_model():
    global model
    model = tf.saved_model.load('saved_model')
    model = model.signatures["serving_default"]

def read_class_labels(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        class_labels = [line.strip().split('/')[-1] for line in file]
    return class_labels

@app.route('/', methods=['GET', 'POST'])
def predict():
    global model, class_labels, predicted_letters

    if model is None:
        load_saved_model()

    if request.method == 'POST':
        action = request.form['action']
        if action == 'Delete':
            if predicted_letters: #check if not empty
                predicted_letters.pop()  
            # After clearing, render the template with the updated prediction
            return render_template('index.html', prediction=''.join(predicted_letters))

      
        image_data = request.form['imagedata']
        try:
            image = base64_to_image(image_data)
            image = image.resize((224, 224))
            image = img_to_array(image)
            image = preprocess_input(image)
            image = np.expand_dims(image, axis=0)

            prediction = model(tf.constant(image))
            output = prediction['output_0']
            predicted_class = tf.argmax(output, axis=1)
            top_class_index = predicted_class[0].numpy()
            top_class_label = class_labels[top_class_index]

            
            predicted_letters.append(top_class_label)
        except Exception as e:
           
            print("Error processing image:", e)

    return render_template('index.html', prediction=''.join(predicted_letters))

def base64_to_image(base64_string):
    try:
        image_data = base64.b64decode(base64_string.split(",")[1])
        image = Image.open(BytesIO(image_data))
        return image
    except Exception as e:
       
        print("Error decoding base64 image data:", e)
        raise

if __name__ == '__main__':
  
    labels_file_path = 'labels.txt'
    class_labels = read_class_labels(labels_file_path)
    
    app.run(port=3000, debug=True)