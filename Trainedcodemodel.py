import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm.notebook import tqdm
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input
from tensorflow.keras.utils import plot_model

# Suppress warnings
warnings.filterwarnings('ignore')

# Define the base directory where the UTKFace dataset is stored
BASE_DIR = '/Users/kavvinaykarthik/Desktop/FaceRecognite/crop_part1'

# Lists to store image paths, age labels, and gender labels
image_paths = []
age_labels = []
gender_labels = []

# Load image paths, age labels, and gender labels from the dataset
for filename in tqdm(os.listdir(BASE_DIR)):
    image_path = os.path.join(BASE_DIR, filename)
    temp = filename.split('_')
    try:
        age = int(temp[0])
        gender = int(temp[1])
        image_paths.append(image_path)
        age_labels.append(age)
        gender_labels.append(gender)
    except ValueError:
        print("Error processing filename:", filename)

# Create a DataFrame to store image paths, age labels, and gender labels
df = pd.DataFrame()
df['image'], df['age'], df['gender'] = image_paths, age_labels, gender_labels

# Display basic statistics and plots
gender_dict = {0: 'Male', 1: 'Female'}
img = Image.open(df['image'][0])
plt.axis('off')
plt.imshow(img)
sns.distplot(df['age'])
sns.countplot(df['gender'])

# Select a random subset of 2500 images for training
subset_df = df.sample(n=8000, random_state=42)

# Feature Extraction
def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, color_mode='grayscale', target_size=(128, 128))
        img = np.array(img)
        features.append(img)

    features = np.array(features)
    features = features.reshape(len(features), 128, 128, 1)
    return features

X = extract_features(subset_df['image'])
X = X / 255.0

y_gender = np.array(subset_df['gender'])
y_age = np.array(subset_df['age'])

input_shape = (128, 128, 1)

inputs = Input((input_shape))
# convolutional layers
conv_1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
maxp_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)
conv_2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(maxp_1)
maxp_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)
conv_3 = Conv2D(128, kernel_size=(3, 3), activation='relu')(maxp_2)
maxp_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)
conv_4 = Conv2D(256, kernel_size=(3, 3), activation='relu')(maxp_3)
maxp_4 = MaxPooling2D(pool_size=(2, 2))(conv_4)

flatten = Flatten()(maxp_4)

# fully connected layers
dense_1 = Dense(256, activation='relu')(flatten)
dense_2 = Dense(256, activation='relu')(flatten)

dropout_1 = Dropout(0.3)(dense_1)
dropout_2 = Dropout(0.3)(dense_2)

output_1 = Dense(1, activation='sigmoid', name='gender_out')(dropout_1)
output_2 = Dense(1, activation='relu', name='age_out')(dropout_2)

model = Model(inputs=[inputs], outputs=[output_1, output_2])

model.compile(loss=['binary_crossentropy', 'mae'], optimizer='adam', metrics=['accuracy'])

# Display the model architecture
plot_model(model)

# Train the model
history = model.fit(x=X, y=[y_gender, y_age], batch_size=30, epochs=30, validation_split=0.2)

# Plot accuracy and loss graphs
plt.plot(history.history['gender_out_accuracy'], label='Gender Accuracy')
plt.plot(history.history['val_gender_out_accuracy'], label='Validation Gender Accuracy')
plt.plot(history.history['age_out_accuracy'], label='Age Accuracy')
plt.plot(history.history['val_age_out_accuracy'], label='Validation Age Accuracy')
plt.legend()
plt.title('Accuracy Graph')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

plt.plot(history.history['gender_out_loss'], label='Gender Loss')
plt.plot(history.history['val_gender_out_loss'], label='Validation Gender Loss')
plt.plot(history.history['age_out_loss'], label='Age Loss')
plt



model.save('age_gender_model2.h5')
