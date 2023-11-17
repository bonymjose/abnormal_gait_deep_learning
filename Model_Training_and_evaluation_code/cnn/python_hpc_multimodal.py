import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, concatenate, Dropout
from tensorflow.keras.metrics import Precision, Recall
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.utils import class_weight

core_path = '/cs/home/psxbj3/deeplearn_ds/'

# Base directory where your dataset is located
dataset_dir = core_path+'/GAIT_IT/'



# Dictionary mapping class names to their respective folder names
class_dirs = {
    'abnormal': ['Diplegic', 'Hemiplegic', 'Neuropathic', 'Parkinsonian'],
    'normal': ['Normal']
}

sample_number_list=[5,10,15,20,25,30.35,40,45,50]
for sample_number in sample_number_list:

    # Lists to store images and labels
    skeleton_images = []
    silhouette_images = []
    labels = []


    # Loop over each class_name and its corresponding directories
    for class_name, patho_folder_names in class_dirs.items():
        for folder in patho_folder_names:
            folder_dir = os.path.join(dataset_dir, folder)
            for participant_number_folder in sorted(os.listdir(folder_dir)):
                if participant_number_folder == '.DS_Store':
                    continue
                participant_number_folder_dir = os.path.join(folder_dir, participant_number_folder)
                               
                # Access the 'skeleton' and 'silhouette' participant_number_folders
                skeleton_sub_dir = os.path.join(participant_number_folder_dir, 'skeletons')
                silhouette_sub_dir = os.path.join(participant_number_folder_dir, 'silhouettes')
                
                if "side_view" in os.listdir(skeleton_sub_dir):
                    skeleton_view_dir = os.path.join(skeleton_sub_dir, "side_view")
                    silhouette_view_dir = os.path.join(silhouette_sub_dir, "side_view")
                   
                else:
                    skeleton_view_dir = skeleton_sub_dir
                    silhouette_view_dir = silhouette_sub_dir
                
                # Iterate over the extra nested folder layer
                for image_folder_name in os.listdir(skeleton_view_dir):
                    if image_folder_name == '.DS_Store' or image_folder_name == 'metadata' or '_front' in image_folder_name or 'lvl2' in image_folder_name:
                        continue
                    image_folder_path = os.path.join(skeleton_view_dir, image_folder_name)
                    silhouette_folder_path = os.path.join(silhouette_view_dir, image_folder_name)
                    print(image_folder_path)
                    print(silhouette_folder_path)
                    
                    skeleton_images_list = sorted(os.listdir(image_folder_path))
                    silhouette_images_list = sorted(os.listdir(silhouette_folder_path))
                    
                    image_counter = 0  # Single counter for both image types
                    
                    for ske_img_name, sil_img_name in zip(skeleton_images_list, silhouette_images_list):
                        if folder in "normal":
                            if image_counter >= 4*sample_number:  # Limit of 4 times sample_number images per folder for "normal"
                                break
                        else:
                            if image_counter >= sample_number:  # Limit of sample_number images per folder for others
                                break
                        
                        # Paths for both image types
                        skeleton_image_path = os.path.join(image_folder_path, ske_img_name)
                        silhouette_image_path = os.path.join(silhouette_folder_path, sil_img_name)

                        print(skeleton_image_path)
                        print(silhouette_image_path)
                        
                        # Load and store skeleton image
                        ske_image = load_img(skeleton_image_path, target_size=(224, 224))
                        ske_image = img_to_array(ske_image)
                        skeleton_images.append(ske_image)
                        
                        # Load and store silhouette image
                        sil_image = load_img(silhouette_image_path, target_size=(224, 224))
                        sil_image = img_to_array(sil_image)
                        silhouette_images.append(sil_image)
                        
                        # Store the label
                        labels.append(list(class_dirs.keys()).index(class_name))
                        
                        # Increment the counter
                        image_counter += 1

    # Convert the lists to NumPy arrays
    skeleton_images = np.array(skeleton_images)
    silhouette_images = np.array(silhouette_images)
    labels = np.array(labels)
  
    # Model architecture
    def dual_input_cnn(input_shape):
        # First input: Skeleton
        skeleton_input = Input(shape=input_shape, name="skeleton_input")
        x1 = Conv2D(32, (3, 3), activation="relu")(skeleton_input)
        x1 = MaxPooling2D((2, 2))(x1)
        x1 = Conv2D(64, (3, 3), activation="relu")(x1)
        x1 = MaxPooling2D((2, 2))(x1)
        x1 = Flatten()(x1)
        
        # Second input: Silhouette
        silhouette_input = Input(shape=input_shape, name="silhouette_input")
        x2 = Conv2D(32, (3, 3), activation="relu")(silhouette_input)
        x2 = MaxPooling2D((2, 2))(x2)
        x2 = Conv2D(64, (3, 3), activation="relu")(x2)
        x2 = MaxPooling2D((2, 2))(x2)
        x2 = Flatten()(x2)
        
        # Merge the outputs of the two branches
        combined = concatenate([x1, x2])
        
        # Add fully connected layers
        z = Dense(128, activation="relu")(combined)
        z = Dropout(0.5)(z) # Regularization using Dropout
        
        # Final output layer - binary classification
        z = Dense(1, activation="sigmoid")(z)
        
        # Create and compile the model
        model = Model(inputs=[skeleton_input, silhouette_input], outputs=z)
        model.compile(optimizer="adam", 
                    loss="binary_crossentropy", 
                    metrics=["accuracy", Precision(name="precision"), Recall(name="recall")])
        
        return model

    input_shape = (224, 224, 3)  
   
    multimodal_path_sample = core_path+'Test-deep_learn/cnn_multi_modal/sample_'+str(sample_number)
    model_save_location = multimodal_path_sample+'/normal_boosted_models/'  # specify the path
    Result_location=multimodal_path_sample+'/normal_boosted_Results/'

    # Ensure the directory exists, if not, create it
    if not os.path.exists(multimodal_path_sample):
        os.makedirs(multimodal_path_sample)

    # Ensure the directory exists, if not, create it
    if not os.path.exists(model_save_location):
        os.makedirs(model_save_location)

    # Ensure the directory exists, if not, create it
    if not os.path.exists(Result_location):
        os.makedirs(Result_location)

    # Define the number of folds
    n_folds = 2
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Prepare lists to store the results for each fold
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    # Initialize the result string
    result_str = ""

    # Combine skel and sil data for easier split
    X_train = [skeleton_images, silhouette_images]
    labels_train = labels

    for fold_number, (train, val) in enumerate(kfold.split(np.zeros(len(labels_train)), labels_train)):
        
        # Create a fresh model for each fold
        model = dual_input_cnn(input_shape)
        
        # Compute class weights for the current split
        weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels_train[train]), y=labels_train[train])
        class_weight_dict = {i: weights[i] for i in np.unique(labels_train[train])}
        
        # Train the model using data from the current split
        model.fit(
            [X_train[0][train], X_train[1][train]], labels_train[train],
            epochs=3, batch_size=32,
            validation_data=([X_train[0][val], X_train[1][val]], labels_train[val]),
            class_weight=class_weight_dict,
            verbose=1  
        )
        
        # Evaluate the model on the validation data of the current split
        loss, accuracy, precision, recall = model.evaluate([X_train[0][val], X_train[1][val]], labels_train[val])
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        # Store the results
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

        raw_predictions = model.predict([X_train[0][val], X_train[1][val]])
        
        # Predict the labels for the validation set
        threshold = 0.5
        predictions = (raw_predictions > threshold).astype(int)
        
        # Calculate the confusion matrix and display it
        matrix = confusion_matrix(labels_train[val], predictions)
        plt.figure()
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for Fold {fold_number + 1}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

        # Save the confusion matrix plot
        plot_filename = os.path.join(Result_location, f'confusion_matrix_fold_{fold_number + 1}.png')
        plt.savefig(plot_filename)

        # Generate the ROC curve
        fpr, tpr, thresholds = roc_curve(labels_train[val], raw_predictions)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=1, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic for Fold {fold_number + 1}')
        plt.legend(loc="lower right")

        # Save the ROC curve
        roc_filename = os.path.join(Result_location, f'roc_curve_fold_{fold_number + 1}.png')
        plt.savefig(roc_filename)
        plt.show()
        print(f"ROC curve for fold {fold_number + 1} saved at {roc_filename}")

        # Append results to the result string
        result_str += f"Fold {fold_number + 1}\n"
        result_str += f"Accuracy: {accuracy*100:.2f}%\n"
        result_str += f"Precision: {precision*100:.2f}%\n"
        result_str += f"Recall: {recall*100:.2f}%\n"
        result_str += f"F1-Score: {f1_score*100:.2f}%\n"
        result_str += f"Confusion Matrix:\n"
        result_str += "\n".join(" ".join(map(str, row)) for row in matrix)
        result_str += "\n\n"

        # Save the trained model for this fold
        model_filename = os.path.join(model_save_location, f'model_fold_{fold_number + 1}.h5')
        model.save(model_filename)
        print(f"Model for fold {fold_number + 1} saved at {model_filename}")

    # Calculate and append average results to the result string
    avg_accuracy = np.mean(accuracies)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1_scores)

    result_str += f"Average Results Across All Folds:\n"
    result_str += f"Average Accuracy: {avg_accuracy*100:.2f}%\n"
    result_str += f"Average Precision: {avg_precision*100:.2f}%\n"
    result_str += f"Average Recall: {avg_recall*100:.2f}%\n"
    result_str += f"Average F1-Score: {avg_f1*100:.2f}%\n"



    # Save to a text file
    with open(Result_location+"kfold_results_sample_"+str(sample_number)+".txt", "w") as file:
        file.write(result_str)

    # Print the average results 
    print(f"Average Accuracy: {avg_accuracy*100:.2f}%")
    print(f"Average Precision: {avg_precision*100:.2f}%")
    print(f"Average Recall: {avg_recall*100:.2f}%")
    print(f"Average F1-Score: {avg_f1*100:.2f}%")



