import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the pre-trained models
model_paths = {
    "Resnet50_ATTENTION Model": 'model_0.1_resnet.h5',
    "INCEPTIONV3_ATTENTION Model": 'INCEPTIONV3_ATTENTION.h5',
    "VGGNET16_ATTENTION Model": 'VGGNET16_ATTENTION.h5'
}

models = {model_name: tf.keras.models.load_model(path) for model_name, path in model_paths.items()}

class_labels = ['Apple healthy', 'Blueberry healthy', 'Cherry (including_sour) healthy', 'Cherry (including sour) Powdery mildew',
                    'Corn (maize) Cercospora leaf spot Gray leaf spot', 'Corn (maize) Common rust','Corn (maize) healthy','Corn (maize) Northern Leaf Blight',
                    'Grape Black rot','Grape Esca (Black Measles)','Grape Leaf blight (Isariopsis Leaf Spot)',
                    'Grape healthy', 'Bell Pepper Bacterial spot', 'Bell Pepper healthy', 'Potato Early blight', 'Potato healthy', 
                'Potato Late blight','Strawberry healthy', 'Strawberry Leaf scorch','Apple Cedar apple rust']  # Replace with your actual class labels

# Or if you have a dictionary mapping class indices to class names
class_label_dict = {0: 'Apple healthy', 1: 'Blueberry healthy', 2: 'Cherry (including_sour) healthy',  3: 'Cherry (including sour) Powdery mildew',
                    4: 'Corn (maize) Cercospora leaf spot Gray leaf spot', 5: 'Corn (maize) Common rust',6: 'Corn (maize) healthy',7: 'Corn (maize) Northern Leaf Blight',
                    8: 'Grape Black rot',9: 'Grape Esca (Black Measles)',10: 'Grape Leaf blight (Isariopsis Leaf Spot)',
                    11: 'Grape healthy', 12: 'Bell Pepper Bacterial spot', 13: 'Bell Pepper healthy', 
                    14: 'Potato Early blight', 15: 'Potato healthy', 16: 'Potato Late blight',
                    17: 'Strawberry healthy',18: 'Strawberry Leaf scorch', 19: 'Apple Cedar apple rust'}  # Replace with your actual class label dictionary

# Define a function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Define a function to predict the class name
def predict_class(image_path, selected_model):
    preprocessed_img = preprocess_image(image_path)
    predictions = selected_model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)
    predicted_class_name = class_label_dict[predicted_class_index[0]]
    return predicted_class_name

# Define a function to handle the upload button click event
def upload_image():
    file_path = filedialog.askopenfilename()
    img = Image.open(file_path)
    img = img.resize((300, 300), Image.LANCZOS)  # Resize image for display
    img = ImageTk.PhotoImage(img)
    panel.configure(image=img)
    panel.image = img  # Keep a reference to the image to prevent garbage collection
    class_name_label.config(text="")  # Clear previous prediction

# Define functions to handle predict button click events for each model
def predict_image_class_model1():
    image_path = filedialog.askopenfilename()
    class_name = predict_class(image_path, models["Resnet50_ATTENTION Model"])
    class_name_label.config(text="Predicted Disease (Resnet50_ATTENTION Model): " + class_name)

def predict_image_class_model2():
    image_path = filedialog.askopenfilename()
    class_name = predict_class(image_path, models["INCEPTIONV3_ATTENTION Model"])
    class_name_label.config(text="Predicted Disease (INCEPTIONV3_ATTENTION Model): " + class_name)

def predict_image_class_model3():
    image_path = filedialog.askopenfilename()
    class_name = predict_class(image_path, models["VGGNET16_ATTENTION Model"])
    class_name_label.config(text="Predicted Disease (VGGNET16_ATTENTION Model): " + class_name)

# Create the main window
root = tk.Tk()
root.title("Plants Disease Detection")
root.configure(bg="black")

# Create and place widgets
upload_button = tk.Button(root, text="Upload Image", command=upload_image, bg="sky blue", fg="black", font=("Elephant", 15), width=70, height=2)
upload_button.pack(pady=10)

panel = tk.Label(root, bg="light blue")
panel.pack(padx=10, pady=10)

# Buttons to predict with each model
predict_button_model1 = tk.Button(root, text="Predict Disease (Resnet50_ATTENTION Model)", command=predict_image_class_model1, bg="lightgreen", fg="black",font=("Elephant", 15), width=70, height=2)
predict_button_model1.pack(pady=5)

predict_button_model2 = tk.Button(root, text="Predict Disease (INCEPTIONV3_ATTENTION Model)", command=predict_image_class_model2, bg="lightcoral", fg="black", font=("Elephant", 15), width=70, height=2)
predict_button_model2.pack(pady=5)

predict_button_model3 = tk.Button(root, text="Predict Disease (VGGNET16_ATTENTION Model)", command=predict_image_class_model3, bg="yellow",fg="black", font=("Elephant", 15), width=70, height=2)
predict_button_model3.pack(pady=5)

class_name_label = tk.Label(root, text="", bg="light blue")
class_name_label.pack(pady=5)

# Run the main event loop
root.mainloop()
