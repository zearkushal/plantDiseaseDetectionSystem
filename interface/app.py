from flask import Flask, render_template, request, redirect, url_for, flash, session
import tensorflow as tf
import numpy as np
import json
import os
import math
import sqlite3
from datetime import datetime, timedelta
import heapq


# Dijkstra's algorithm function
def dijkstra(graph, start, end):
    queue = []
    heapq.heappush(queue, (0, start))  # (cumulative_cost, node)
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    visited = set()
    
    while queue:
        current_distance, current_node = heapq.heappop(queue)
        
        if current_node in visited:
            continue
        visited.add(current_node)
        
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(queue, (distance, neighbor))
    
    return distances[end]

# Graph of locations (Chabahil is fixed, others are dynamic)
locations = {
        'Chabahil': {'Gaushala': 5, 'Sinamagal': 10},
        'Gaushala': {'Chabahil': 5, 'Baneshwor': 2},
        'Sinamagal': {'Chabahil': 10, 'Baneshwor': 4, 'Thapathali': 8},
        'Baneshwor': {'Gaushala': 2, 'Sinamagal': 4, 'Thapathali': 3},
        'Thapathali': {'Sinamagal': 8, 'Baneshwor': 3}
}
app = Flask(__name__)
app.secret_key = 'darumaikki'  # Required for session and flash messages

# Load the training history from the JSON file
with open('../training_histtory.json', 'r') as f:
    history = json.load(f)

# Load the disease descriptions
with open('disease_descriptions.json', 'r') as f:
    disease_descriptions = json.load(f)

# Load the disease supplements
with open('disease_supplements.json', 'r') as f:
    disease_to_supplement = json.load(f)

# Load the disease medicine
with open('disease_medicine.json', 'r') as f:
    disease_to_medicine = json.load(f)

# Load class names
with open('class_name.json', 'r') as f:
    class_name = json.load(f)

# Initialize SQLite DB for login/signup
def init_db():
    conn = sqlite3.connect('diseasedetection.db')  # Use the correct database name
    cursor = conn.cursor()
    
    # Create 'users' and 'logins' tables
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL,
                        email TEXT UNIQUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS logins (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        password INTEGER NOT NULL,
                        login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        logout_time TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users(id)
                    )''')
    
    conn.commit()
    conn.close()

init_db()

# Function to check if a user exists
def user_exists(username):
    conn = sqlite3.connect('diseasedetection.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()
    return user

# Function to add a new user
def add_user(username, password, email):
    conn = sqlite3.connect('diseasedetection.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO users (username, password, email) VALUES (?, ?, ?)', (username, password, email))
    conn.commit()
    conn.close()

# Function to log user login time
def log_user_login(user_id):
    conn = sqlite3.connect('diseasedetection.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO logins (user_id) VALUES (?)', (user_id,))
    conn.commit()
    conn.close()

# Function to log user logout time
def log_user_logout(user_id):
    conn = sqlite3.connect('diseasedetection.db')
    cursor = conn.cursor()
    cursor.execute('UPDATE logins SET logout_time = ? WHERE user_id = ? AND logout_time IS NULL', 
                   (datetime.now(), user_id))
    conn.commit()
    conn.close()

# TensorFlow Model Prediction with Confidence
def model_prediction(test_image_path):
    model = tf.keras.models.load_model("../trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    confidence = np.max(predictions)  # Confidence score
    return np.argmax(predictions), confidence  # Return index and confidence

@app.route('/home', methods=['GET', 'POST'])
def home():
    prediction = None
    supplement = None
    disease_info = None
    medicine = None
    confidence = None
    file_path = None
    accuracy = history['accuracy'][-1]  # Latest training accuracy

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('home.html', accuracy=accuracy)

        file = request.files['file']
        if file.filename == '':
            return render_template('home.html', accuracy=accuracy)

        if file:
            # Ensure the images directory exists
            images_dir = os.path.join('static', 'images')
            os.makedirs(images_dir, exist_ok=True)

            # Save the file in the static/images directory
            file_name = file.filename
            file_path = os.path.join('images/', file_name)
            file.save(os.path.join('static', file_path))

            # Predict
            result_index, confidence = model_prediction(os.path.join('static', file_path))

            if confidence < 0.5:  # Low confidence
                prediction = "Unknown or Uncertain"
                supplement = "N/A"
                disease_info = "The model is not confident about this image."
                medicine = "No medicine recommendation available."
            else:
                predicted_class = class_name[result_index]
                supplement_info = disease_to_supplement.get(predicted_class, "No supplement information available.")
                disease_info = disease_descriptions.get(predicted_class, "The Description is unavailable.")
                medicine = disease_to_medicine.get(predicted_class, "No medicine information available.")
                prediction = predicted_class
                supplement = supplement_info

    return render_template('home.html', prediction=prediction, supplement=supplement, diseaseDescription=disease_info, medicine=medicine, file_path=file_path, confidence=confidence, accuracy=accuracy)

# @app.route("/home",methods=['GET', 'POST'])
# def get_medicine():
#     estimated_time = None
#     start_location = None
#     end_location = 'Chabahil'  # fixed to Chabahil for simplicity

#     if request.method == 'POST':
#         start_location = request.form.get('start_location')
#         if start_location:
#         # Call Dijkstra's algorithm to calculate the estimated time
#             estimated_time = dijkstra(locations, start_location, end_location)
#     return render_template('home.html', locations=locations, estimated_time=estimated_time, start=start_location, end=end_location)
#below code works. yo backup hunxa
@app.route("/algodemo",methods=['GET', 'POST'])
def algodemo():
    estimated_time = None
    start_location = None
    end_location = 'Chabahil'  # fixed to Chabahil for simplicity
    delivery_time = None

    if request.method == 'POST':
        start_location = request.form.get('start_location')
        if start_location:
        # Call Dijkstra's algorithm to calculate the estimated time
            estimated_time = dijkstra(locations, start_location, end_location)
            if estimated_time is not None:
                current_time = datetime.now()
                delivery_time = current_time + timedelta(minutes=estimated_time)

    return render_template('algodemo.html', locations=locations, estimated_time=estimated_time, start=start_location, end=end_location,delivery_time=delivery_time.strftime('%I:%M %p') if delivery_time else None
)

## Landing Page
@app.route("/")
def default():
    return render_template('first.html')

## About Section
@app.route("/about")
def about():
    return render_template('about.html')

## Contact Us Section
@app.route("/contact")
def contact():
    return render_template('contact.html')

@app.route("/navbar")
def navbar():
    return render_template('navbar.html')

@app.route("/demologin")
def demologin():
    return render_template('demologin.html')

@app.route("/testingxa")
def testingxa():
    return render_template('testingxa.html')

@app.route("/signup", methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['signupusername']
        password = request.form['signuppassword']
        email = request.form['signupemail']
        
        if user_exists(username):
            flash('Username already exists, please choose a different one.', 'danger')
            return redirect(url_for('signup'))

        add_user(username, password, email)
        flash('Signup successful! You can now log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = user_exists(username)
        if user and user[2] == password:  # Check password from the 'users' table
            # Log the login time
            log_user_login(user[0])  # Pass the user_id (user[0]) from the 'users' table
            session['user_id'] = user[0]  # Store user_id in the session
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password.', 'danger')

    return render_template('login.html')

@app.route("/logout")
def logout():
    user_id = session.get('user_id')
    if user_id:
        log_user_logout(user_id)
        session.clear()
        flash('Logged out successfully.', 'success')
    return redirect(url_for('login'))

@app.route("/test")
def test():
    # Pagination setup
    page = int(request.args.get('page', 1))
    images_per_page = 9  # Total images per page
    images_per_row = 3  # Images per row
    start_index = (page - 1) * images_per_page
    end_index = start_index + images_per_page

    # List all image files in the 'test' directory
    images_dir = os.path.join('static', 'test')
    images = sorted(os.listdir(images_dir))

    # Slice the images list for the current page
    images = images[start_index:end_index]

    # Calculate the total number of pages
    total_images = len(os.listdir(images_dir))
    total_pages = math.ceil(total_images / images_per_page)
    return render_template('testImage.html', images=images, current_page=page, total_pages=total_pages)

if __name__ == '__main__':
    app.run(debug=True, port=8000)
