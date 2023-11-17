import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, concatenate, Dropout
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Reshape, Dense, Flatten
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

sample_number_list=[5,10,15,20,25,30,35,40,45,50]
for sample_number in sample_number_list:
    # Lists to store images and labels
    skeleton_images = []
    labels = []
    sequence_length = sample_number                                                     


    # Loop over each class_name and its corresponding directories
    for class_name, patho_folder_names in class_dirs.items():
        for folder in patho_folder_names:
            folder_dir = os.path.join(dataset_dir, folder)
            for participant_number_folder in sorted(os.listdir(folder_dir)):
                if participant_number_folder == '.DS_Store':
                    continue
                participant_number_folder_dir = os.path.join(folder_dir, participant_number_folder)
                
                skeleton_sub_dir = os.path.join(participant_number_folder_dir, 'skeletons')
                
                if "side_view" in os.listdir(skeleton_sub_dir):
                    skeleton_view_dir = os.path.join(skeleton_sub_dir, "side_view")
                else:
                    skeleton_view_dir = skeleton_sub_dir
                
                # Iterate over the extra nested folder layer
                for image_folder_name in os.listdir(skeleton_view_dir):
                    if image_folder_name == '.DS_Store' or image_folder_name == 'metadata' or '_front' in image_folder_name or 'lvl2' in image_folder_name:
                        continue
                    image_folder_path = os.path.join(skeleton_view_dir, image_folder_name)
                    print(image_folder_path)
                    
                    skeleton_images_list = sorted(os.listdir(image_folder_path))
                    
                    image_counter = 0  # counter
                    current_sequence = []                                               

                    for ske_img_name in skeleton_images_list:
                        if folder in "normal":
                            if image_counter >= 4*sample_number:  # Limit of 4 times sample_number images per folder for "normal"
                                break
                        else:
                            if image_counter >= sample_number:  # Limit of sample_number images per folder for others
                                break
                        
                        # Paths for image types
                        skeleton_image_path = os.path.join(image_folder_path, ske_img_name)

                        print(skeleton_image_path)
                                                
                        # Load and store skeleton image
                        ske_image = load_img(skeleton_image_path, target_size=(224, 224))
                        ske_image = img_to_array(ske_image)
                        current_sequence.append(ske_image)                              
                        
                        if len(current_sequence) == sequence_length:                    
                            skeleton_images.append(np.array(current_sequence))          
                            labels.append(list(class_dirs.keys()).index(class_name))    
                            current_sequence = []  # Reset the current sequence         
                        
                        # Increment the counter
                        image_counter += 1

    # Convert the lists to NumPy arrays
    skeleton_images = np.array(skeleton_images)
    labels = np.array(labels)

    print(skeleton_images.shape)                                                        
    print(labels.shape)                                                                


    # Model architecture
    def skeletal_input_lstm(input_shape):
        # Input layer
        skeleton_input = Input(shape=input_shape, name='skeleton_input')

        # Reshape layer
        x = Reshape((-1, input_shape[1]*input_shape[2]*input_shape[3]))(skeleton_input)

        # LSTM layers
        x = LSTM(128, return_sequences=True)(x)
        x = LSTM(64)(x)

        # Flatten layer
        x = Flatten()(x)

        # Output layer for binary classification
        output = Dense(units=1, activation='sigmoid')(x)

        # Creating the Model
        model = Model(inputs=skeleton_input, outputs=output)

        # Compile the model for binary classification
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy", Precision(name="precision"), Recall(name="recall")])
        
        return model

    # Update input_shape variable and model function call
    input_shape = (sample_number, 224, 224, 3)  

        
    skeletal_path_sample = core_path+'Test-deep_learn/lstm_skeletal/sample_'+str(sample_number)
    model_save_location = skeletal_path_sample+'/normal_boosted_models/'  # specify the path
    Result_location=skeletal_path_sample+'/normal_boosted_Results/'

    # Ensure the directory exists, if not, create it
    if not os.path.exists(skeletal_path_sample):
        os.makedirs(skeletal_path_sample)

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

    X_train = skeleton_images
    labels_train = labels

    for fold_number, (train, val) in enumerate(kfold.split(np.zeros(len(labels_train)), labels_train)):
        
        # Create a fresh model for each fold
        model = skeletal_input_lstm(input_shape)
        model.summary()
        
        # Compute class weights for the current split
        weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels_train[train]), y=labels_train[train])
        class_weight_dict = {i: weights[i] for i in np.unique(labels_train[train])}
        
        # Train the model using data from the current split
        model.fit(
            X_train[train], labels_train[train],
            epochs=3, batch_size=32,
            validation_data=(X_train[val], labels_train[val]),
            class_weight=class_weight_dict,
            verbose=1  
        )
        
        # Evaluate the model on the validation data of the current split
        loss, accuracy, precision, recall = model.evaluate(X_train[val], labels_train[val])
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        # Store the results
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

        raw_predictions = model.predict(X_train[val])
        
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

        # Save the confusion matrix plot
        plot_filename = os.path.join(Result_location, f'confusion_matrix_fold_{fold_number + 1}.png') 
        plt.savefig(plot_filename)                                                                     
        print(f"Confusion matrix for fold {fold_number + 1} saved at {plot_filename}")              
        plt.show()                                                                                

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



