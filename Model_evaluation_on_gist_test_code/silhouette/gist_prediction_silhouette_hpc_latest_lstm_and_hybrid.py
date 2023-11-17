import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


image_count_limit_list=[5,10,15,20,25,30,35,40,45,50]

for image_count_limit in image_count_limit_list:

    core_path = '/cs/home/psxbj3/deeplearn_ds/'

    # Dictionary mapping class names to their respective folder names
    class_dirs = {
        'abnormal': ['Diplegic', 'Hemiplegic', 'Neuropathic', 'Parkinsonian'],
        'normal': ['Normal']
    }

    # Base directory where the unknown dataset is located
    unknown_dataset_dir = core_path+'GAIT_IST/'

    # Lists to store images and labels
    silhouette_images = []
    labels = []
    sequence_length = image_count_limit

    # Loop over each class_name and its corresponding directories
    for class_name, patho_folder_names in class_dirs.items():
        for folder in patho_folder_names:
            folder_dir = os.path.join(unknown_dataset_dir, folder)
            for participant_number_folder in sorted(os.listdir(folder_dir)):
                if participant_number_folder == '.DS_Store':
                    continue
                participant_number_folder_dir = os.path.join(folder_dir, participant_number_folder)
                silhouette_sub_dir = os.path.join(participant_number_folder_dir, 'silhouettes')
                
                if "side_view" in os.listdir(silhouette_sub_dir):
                    silhouette_view_dir = os.path.join(silhouette_sub_dir, "side_view")
                else:
                    silhouette_view_dir = silhouette_sub_dir
                
                # Iterate over the extra nested folder layer
                for image_folder_name in os.listdir(silhouette_view_dir):
                    if image_folder_name == '.DS_Store' or image_folder_name == 'metadata' or '_front' in image_folder_name or 'lvl2' in image_folder_name:
                        continue
                    image_folder_path = os.path.join(silhouette_view_dir, image_folder_name)
                    
                    silhouette_images_list = sorted(os.listdir(image_folder_path))
                    
                    image_counter = 0  # counter 
                    current_sequence = []   
                    
                    for sil_img_name in silhouette_images_list:
                        if folder in "normal":
                            if image_counter >= 4*image_count_limit:  # Limit of 4 times image_count_limit images per folder for "normal"
                                break
                        else:
                            if image_counter >= image_count_limit:  # Limit of image_count_limit images per folder for others
                                break
                        
                        silhouette_image_path = os.path.join(image_folder_path, sil_img_name)
                        print(silhouette_image_path)

                        
                        # Load and store silhouette image
                        sil_image = load_img(silhouette_image_path, target_size=(224, 224))
                        sil_image = img_to_array(sil_image)
                        current_sequence.append(sil_image)

                        if len(current_sequence) == sequence_length:                    
                            silhouette_images.append(np.array(current_sequence))          
                            labels.append(list(class_dirs.keys()).index(class_name))    
                            current_sequence = []  # Reset the current sequence   
                        
                        # Increment the counter
                        image_counter += 1

    # Convert the lists to NumPy arrays
    silhouette_images = np.array(silhouette_images)
    labels = np.array(labels)

    print(silhouette_images.shape)                                                        
    print(labels.shape)    

    for dl_type in ['lstm','hybrid']:

        silhouette_path = core_path+'Test-deep_learn/'+dl_type+'_silhouette/'
        for sample_folder in os.listdir(silhouette_path):
            if sample_folder == '.DS_Store':
                continue
            if sample_folder !='sample_'+str(image_count_limit):
                continue
            sample_folder_path = os.path.join(silhouette_path, sample_folder)
            model_dir = os.path.join(sample_folder_path, 'normal_boosted_models')
            result_dir = os.path.join(sample_folder_path, 'normal_boosted_Results')
            result_file = os.path.join(result_dir, 'gist_img_counter_'+str(image_count_limit)+'.txt')
            with open(result_file, "w") as f:
                f.write(f"Sample Folder: {sample_folder}\n")
                # Iterate over each sample folder 
                for model_name in os.listdir(model_dir):
                    if model_name.endswith('.h5'):
                        model_path = os.path.join(model_dir, model_name)
                        
                        # Load the model
                        model = load_model(model_path)

                        # Predict using the model
                        predicted_probs = model.predict(silhouette_images)
                        predicted_class_indices = (predicted_probs > 0.5).astype(int).flatten()

                        # Compute metrics
                        accuracy = accuracy_score(labels, predicted_class_indices)
                        precision = precision_score(labels, predicted_class_indices)
                        recall = recall_score(labels, predicted_class_indices)
                        f1 = f1_score(labels, predicted_class_indices)

                        # Confusion Matrix
                        matrix = confusion_matrix(labels, predicted_class_indices)

                        # Append results for this model to the text file

                        f.write(f"Results for Model: {model_name}\n")
                        f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
                        f.write(f"Precision: {precision * 100:.2f}%\n")
                        f.write(f"Recall: {recall * 100:.2f}%\n")
                        f.write(f"F1-Score: {f1 * 100:.2f}%\n")
                        f.write("\nConfusion Matrix:\n")
                        f.write(str(matrix))
                        f.write("\n=========================\n\n")

                        # Optional: Plot confusion matrix for each model
                        plt.figure(figsize=(10,7))
                        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
                        plt.title(f'Confusion Matrix for {model_name}')
                        plt.xlabel('Predicted Label')
                        plt.ylabel('True Label')


                        # Save the confusion matrix plot
                        plot_filename = os.path.join(result_dir, f'gist_confusion_matrix_model_name_{model_name}.png')
                        plt.savefig(plot_filename)
                        plt.show()
                        print(f"GIST Confusion matrix for fold {model_name} saved at {plot_filename}")

                        # Generate the ROC curve
                        fpr, tpr, thresholds = roc_curve(labels, predicted_probs)
                        roc_auc = auc(fpr, tpr)
                        
                        plt.figure()
                        plt.plot(fpr, tpr, color='darkorange', lw=1, label=f'ROC curve (area = {roc_auc:.2f})')
                        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
                        plt.xlim([0.0, 1.0])
                        plt.ylim([0.0, 1.05])
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        plt.title(f'Receiver Operating Characteristic for Model name {model_name}')
                        plt.legend(loc="lower right")

                        # Save the ROC curve
                        roc_filename = os.path.join(result_dir, f'gist_roc_curve_fold_{model_name}.png')
                        plt.savefig(roc_filename)
                        plt.show()
                        print(f"GIST ROC curve for Model name {model_name} saved at {roc_filename}")

                        print(f"Sample Folder: {sample_folder}\n")
                        print(f"Results for Model: {model_name}")
                        print(f"Accuracy: {accuracy * 100:.2f}%")
                        print(f"Precision: {precision * 100:.2f}%")
                        print(f"Recall: {recall * 100:.2f}%")
                        print(f"F1-Score: {f1 * 100:.2f}%")
                        print("\nConfusion Matrix:")
                        print(matrix)
                        print("\n=========================\n")

