from flask import Flask, request, jsonify
import tensorflow as tf
import cv2
from skimage import io
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

def load_image_model():
    best_model = load_model('./models/classify_model.h5')
    return best_model

def predict_image(img_url, model):
    img = io.imread(img_url)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(img, (224, 224))
    img_tensor = tf.convert_to_tensor(resized_img, dtype=tf.float32)
    img_tensor = tf.expand_dims(img_tensor, 0)
    prediction = model.predict(img_tensor, use_multiprocessing=True)
    prediction = prediction.argmax()
    return prediction

@app.route('/')
def hello():
    if 'image_url' in request.args:
        image_url = request.args['image_url']
        model = load_image_model()
        prediction = predict_image(image_url, model)
        print(prediction)
        return jsonify({"food_prediction": food_array[prediction]})
    else:
        return "Please provide an 'image_url' parameter."

if __name__ == '__main__':
   food_array = [
    "Chicken Green Curry",
    "Pork Curry with Morning Glory",
    "Spicy mixed vegetable soup",
    "Pork Chopped Tofu Soup",
    "Stuffed Bitter Gourd Broth",
    "Chicken Mussaman Curry",
    "Sour Soup",
    "Stir Fried Chicken with Chestnuts",
    "Omelet",
    "Fried egg",
    "Egg and Pork in Sweet Brown Sauce",
    "Egg with Tamarind Sauce",
    "Banana in coconut milk",
    "Stir Fried Rice Noodles with Chicken",
    "Fried Cabbage with Fish Sauce",
    "Grilled River Prawn",
    "Baked Prawns With Vermicelli",
    "Coconut rice pancake",
    "Mango Sticky Rice",
    "Thai Pork Leg Stew",
    "Shrimp Paste Fried Rice",
    "Curried Noodle Soup with Chicken",
    "Fried rice",
    "Shrimp Fried Rice",
    "Steamed capon in flavored rice",
    "Thai Chicken Biryani",
    "Thai Chicken Coconut Soup",
    "River prawn spicy soup",
    "Fried fish-paste balls",
    "Deep fried spring roll",
    "Stir-Fried Chinese Morning Glory",
    "Fried noodle Thai style with prawns",
    "Stir fried Thai basil with minced pork",
    "Fried Noodle in Soy Sauce",
    "Stir-fried Pumpkin with Eggs",
    "Stir-Fried Eggplant with Soybean Paste Sauce",
    "Stir Fried Clams with Roasted Chili Paste",
    "Golden Egg Yolk Threads",
    "Chicken Panang",
    "Thai Wing Beans Salad",
    "Spicy Glass Noodle Salad",
    "Spicy minced pork salad",
    "Egg custard in pumpkin",
    "Tapioca Balls with Pork Filling",
    "Green Papaya Salad",
    "Thai-Style Grilled Pork Skewers",
    "Pork Satay with Peanut Sauce",
    "Steamed Fish with Curry Paste"
    ]   
app.run(debug=True, port=3000)
