import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from scipy.cluster.vq import vq
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Constant
IMG_SIZE = 128

# Adjustable features

N_FEATURES = 100  # Number of features for K means, 
N_CLASSES = 3  # COVID, NonCOVID, Adecarcinoma (if we have more data, ex: pneumonia, we can another class)
MAX_IMAGES = 150 # How many do you want to test?

# Program is formatted with used functions first, driver code on the bottom

# Function to extract SIFT features from a list of images
def extract_sift_features(image_paths):
    sift = cv2.SIFT_create()
    descriptors_list = []
    for image_path in image_paths:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue  # Skip this image and move to the next
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        keypoints, descriptors = sift.detectAndCompute(img, None)
        if descriptors is not None:
            descriptors_list.append(descriptors)
    return descriptors_list

# Function to convert the list of SIFT descriptors for each image into a histogram
def convert_features_histogram(descriptors_list, kmeans):
    histograms = []
    for descriptors in descriptors_list:
        histogram, _ = vq(descriptors, kmeans.cluster_centers_)
        histograms.append(np.bincount(histogram, minlength=N_FEATURES))
    return histograms

# Function to plot the histogram of visual words for an image
def plot_histogram(histogram, title):
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(histogram)), histogram, width=1)
    plt.title(title)
    plt.xlabel('Visual Word Index')
    plt.ylabel('Frequency')
    plt.show()

def overlay_predictions(image_paths, predictions, output_path):
    for i, image_path in enumerate(image_paths):
        img = cv2.imread(image_path)
        label = 'COVID' if predictions[i] == 1 else 'Non-COVID'
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imwrite(os.path.join(output_path, f'prediction_{i}.png'), img)


# Function to create a Bag of Visual Words (BoVW)
def create_bovw(descriptors_list, n_clusters):
    
    filtered_descriptors = [desc for desc in descriptors_list if desc is not None]
    
    if not filtered_descriptors:
        raise ValueError("No descriptors found. Check the images or SIFT detection parameters.")
    
    
    all_descriptors = np.vstack(filtered_descriptors)
    
    # kmeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(all_descriptors)
    return kmeans


# Visualization Functions Below

def visualize_clusters_on_images(image_paths, kmeans, n_clusters):
    sift = cv2.SIFT_create()
    
    
    n_rows = 2
    n_cols = 5

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 8))
    
    axes = axes.flatten()
    
    for ax, image_path in zip(axes, image_paths[:10]):
    
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        keypoints, descriptors = sift.detectAndCompute(img, None)

        if descriptors is not None:
            
            labels = kmeans.predict(descriptors)

            output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for kp, label in zip(keypoints, labels):
                color = plt.cm.jet(label / n_clusters) 
                color = (color[2] * 255, color[1] * 255, color[0] * 255) 
                cv2.circle(output_img, (int(kp.pt[0]), int(kp.pt[1])), 3, color, -1)


            ax.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{os.path.basename(image_path)}")
            ax.axis('off')
        else:
            ax.axis('off')  


    plt.tight_layout()
    plt.show()

def visualize_side_by_side(image_path_1, image_path_2, kmeans, n_clusters):
    sift = cv2.SIFT_create()
    

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    for idx, image_path in enumerate([image_path_1, image_path_2]):

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        keypoints, descriptors = sift.detectAndCompute(img, None)
        ax = ax1 if idx == 0 else ax2

        if descriptors is not None:

            labels = kmeans.predict(descriptors)
            output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for kp, label in zip(keypoints, labels):
                color = plt.cm.jet(label / n_clusters)  
                color = (color[2] * 255, color[1] * 255, color[0] * 255)  
                cv2.circle(output_img, (int(kp.pt[0]), int(kp.pt[1])), 3, color, -1)

            ax.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
            title = 'COVID Image' if idx == 0 else 'Non-COVID Image'
            ax.set_title(f"{title} - {os.path.basename(image_path)}")
            ax.axis('off')

    plt.show()


# Driver Code

# Paths to your data, if you add a class add a path, and add additions to each code block below
base_path = '/Users/ryanhavens/Desktop'
covid_path = os.path.join(base_path, '2COVID')
non_covid_path = os.path.join(base_path, '1NonCOVID')
ade_path = os.path.join(base_path,'adenocarcinoma')

# Get the list of image paths
covid_image_paths = [os.path.join(covid_path, f) for f in os.listdir(covid_path) if f.endswith('.png')]
non_covid_image_paths = [os.path.join(non_covid_path, f) for f in os.listdir(non_covid_path) if f.endswith('.png')]
ade_image_paths = [os.path.join(ade_path, f) for f in os.listdir(ade_path) if f.endswith('.png')]

# Referencing our Max Image variable for how many we want tested
covid_image_paths = covid_image_paths[:MAX_IMAGES]
non_covid_image_paths = non_covid_image_paths[:MAX_IMAGES]
ade_image_paths = ade_image_paths[:MAX_IMAGES]

# Extract SIFT features from all images
covid_descriptors = extract_sift_features(covid_image_paths)
non_covid_descriptors = extract_sift_features(non_covid_image_paths)
ade_descriptors = extract_sift_features(ade_image_paths)

# Combine all descriptors in a list
all_descriptors = covid_descriptors + non_covid_descriptors + ade_descriptors

# Create the BoVW model using KMeans
kmeans = create_bovw(all_descriptors, N_FEATURES)

# Create histograms, add new class here
covid_histograms = convert_features_histogram(covid_descriptors, kmeans)
non_covid_histograms = convert_features_histogram(non_covid_descriptors, kmeans)
ade_histograms = convert_features_histogram(ade_descriptors, kmeans)

# Plotting histograms for the first image of each class. Can be modified to desired amount
for i in range(min(1, len(covid_histograms))):  # Just plot the first 10 for brevity
   plot_histogram(covid_histograms[i], f'COVID Image {i+1} Histogram of Visual Words')
for i in range(min(1, len(non_covid_histograms))):  # Just plot the first 10 for brevity
    plot_histogram(non_covid_histograms[i], f'Non-COVID Image {i+1} Histogram of Visual Words')
for i in range(min(1, len(ade_histograms))):  # Just plot the first 10 for brevity
    plot_histogram(ade_histograms[i], f'Adenocarcinoma Image {i+1} Histogram of Visual Words')

# Create labels, add new label for new class here
labels = np.concatenate([
    np.ones(len(covid_histograms)),  # COVID
    np.zeros(len(non_covid_histograms)),  # non-COVID
    np.full(len(ade_histograms), 2)  # adenocarcinoma
])

# Prepare training data, add new class to our X list
X = np.vstack([non_covid_histograms, covid_histograms, ade_histograms])
y = labels

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train a classifier, in this case, SVM
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Combine COVID and non-COVID image paths into a single list
combined_image_paths = covid_image_paths + non_covid_image_paths + ade_image_paths
visualize_clusters_on_images(combined_image_paths, kmeans, N_FEATURES)  # N_FEATURES is your number of clusters

# Specify which image to display, I picked the first since it looks nice
covid_image_for_comparison = covid_image_paths[1]  
# Specify which image to display, I picked the fourth since it looks nice
non_covid_image_for_comparison = non_covid_image_paths[4]  
visualize_side_by_side(covid_image_for_comparison, non_covid_image_for_comparison, kmeans, N_FEATURES)