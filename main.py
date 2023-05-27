# Tools
import os
import pickle
from typing import Dict
import seaborn as sns
import pandas as pd
from nltk.tokenize import word_tokenize
import numpy as np
import matplotlib.pyplot as plt

# Evaluation
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, fowlkes_mallows_score, jaccard_score, \
    silhouette_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Models
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def save_pickle(obj, file_path):
    """
    Save an object as a pickle file

    :param obj: The object to be saved
    :param file_path: The file path where the object will be saved
    """
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(file_path):
    """
    Load an object from a pickle file.

    :param file_path: The file path of the pickle file.
    :return: The loaded object
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def read_data(file_path):
    """
        Reads the data from the given file path.

        Args:
            file_path (str): Path to the input file.

        Returns:
            pandas.DataFrame: Data in a pandas DataFrame.
    """
    with open(file_path, 'r') as file:
        data = pd.read_csv(file, skip_blank_lines=False)

    return data


def merge_data(data1, data2, left_on, right_on):
    """
    Merges two data sets along a similar column

    Args:
        data1 (pandas.DataFrame): First data set.
        data2 (pandas.DataFrame): Second data set.
        left_on (str) : column name
        right_on (str) : column name

    Returns:
        pandas.DataFrame: Merged data set.
    """
    merged_data = pd.merge(data1, data2, left_on=left_on, right_on=right_on)
    return merged_data


def generate_kmers(df: pd.DataFrame, config: Dict[str, int]) -> pd.DataFrame:
    """
    Generate k-mers from CDR3 sequences in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing CDR3 sequences.
        config (Dict[str, int]): Configuration dictionary with the following keys:
            - cdr3_column: The column name with the CDR3 sequences.
            - non_overlapping_k: k-value for non-overlapping processing.
            - overlapping_k: k-value for overlapping processing.
            - use_non-overlapping : Whether to apply non-overlapping processing (boolean).
            - use_overlapping: Whether to apply overlapping processing (boolean).

    Returns:
        pd.DataFrame: A DataFrame with new columns containing non-overlapping and/or overlapping k-mers.
    """

    def non_overlapping_processing(sequence, k):
        kmers = [sequence[i:i + k] for i in range(0, len(sequence), k)]
        new_sequences = []
        for i in range(k):
            new_sequence = " ".join(kmers[i::k])
            new_sequences.append(new_sequence)
        return new_sequences

    def overlapping_processing(sequence, k):
        kmers = [sequence[i:i + k] for i in range(len(sequence) - k + 1)]
        new_sequence = " ".join(kmers)
        return [new_sequence]

    cdr3_column = config['cdr3_column']
    non_overlapping_k = config['non_overlapping_k']
    overlapping_k = config['overlapping_k']
    use_nonoverlapping = config['use_non-overlapping']
    use_overlapping = config['use_overlapping']

    if use_nonoverlapping:
        df['non_overlapping_sequences'] = df[cdr3_column].apply(non_overlapping_processing, k=non_overlapping_k)

    if use_overlapping:
        df['overlapping_sequences'] = df[cdr3_column].apply(overlapping_processing, k=overlapping_k)

    return df


def create_corpus(df: pd.DataFrame, column_name: str, corpus: list) -> list:
    """
    Create a corpus from the k-mer sequences in the specified column of the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing k-mer sequences.
        column_name (str): The name of the column containing k-mer sequences.
        corpus (list): An empty list to store the corpus.

    Returns:
        list: The updated corpus list.
    """

    for seq in df[column_name]:
        for kmer in seq:
            tokens = word_tokenize(kmer)
            corpus.append(TaggedDocument(tokens, [len(corpus)]))

    return corpus


def train_and_save_vectors(corpus: list, df: pd.DataFrame, output_filename: str) -> pd.DataFrame:
    """
    Train a Doc2Vec model using the input corpus, create vectors for each sequence,
    update the input DataFrame with the vectors, and save the updated DataFrame to a CSV file.

    Args:
        corpus (list): A list of TaggedDocument objects representing the k-mer sequences.
        df (pd.DataFrame): The input DataFrame to update with vector representations.
        output_filename (str): The filename to save the updated DataFrame as a CSV file.

    Returns:
        pd.DataFrame: The updated DataFrame with vector representations.
    """
    # Train the Doc2Vec model
    model = Doc2Vec(corpus, vector_size=100, window=25, min_count=2, workers=4, epochs=20)

    # Get the vector representation of each sequence
    vectors = []
    for i in range(len(corpus)):
        vector = model.infer_vector(corpus[i].words)
        vectors.append(vector)

    # Check if the length of vectors matches the number of rows in the DataFrame
    if len(vectors) != len(df):
        print('The length of vectors does not match the number of rows in the DataFrame.')
        # Add NaN values to the DataFrame for missing vectors
        if len(vectors) < len(df):
            num_missing = len(df) - len(vectors)
            for i in range(num_missing):
                vectors.append([np.nan] * len(vectors[0]))
        # Remove extra vectors from the list
        else:
            vectors = vectors[:len(df)]

    # Add the vector representations as new columns in the data frame
    vector_columns = ['vector_' + str(i + 1) for i in range(len(vectors[0]))]
    for i in range(len(vectors[0])):
        df[vector_columns[i]] = [vector[i] for vector in vectors]

    # Save the updated data frame to a new CSV file
    df.to_csv(output_filename)

    return df


def apply_pca(df, columns, n_components=None):
    """
    Apply PCA to selected columns and return transformed data.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): A list of column names to apply PCA on.
        n_components (int, optional): Number of components to keep.

    Returns:
        np.ndarray: The transformed data after applying PCA.
    """
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(df[columns])
    return transformed_data


def train_model(data, model_type, **kwargs):
    """
    Trains a model on the given data.

    Args:
        data (pandas.DataFrame): Data to train the model on.
        model_type (str): Type of the model to train ('kmeans', 'hierarchical', 'dbscan', 'gmm', etc.).
        **kwargs: Optional keyword arguments for the chosen model type.

    Returns:
        object: Trained model instance.
    """
    if model_type == 'kmeans':
        model = KMeans(**kwargs)
    elif model_type == 'hierarchical':
        model = AgglomerativeClustering(**kwargs)
    elif model_type == 'dbscan':
        model = DBSCAN(**kwargs)
    elif model_type == 'gmm':
        model = GaussianMixture(**kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    if model_type == 'gmm':
        model.fit(data)
        model.labels_ = model.predict(data)
    else:
        model.fit(data)

    return model


def evaluate_model(model, data, labels, metric):
    """
    Evaluates the trained model using the specified metric.

    Args:
        model (object): Trained model instance.
        data (pandas.DataFrame): Data to evaluate the model on.
        labels (array): Ground truth labels.
        metric (str): Evaluation metric ('silhouette_score', 'ari', 'nmi', 'fmi', 'jaccard', etc.).

    Returns:
        float: Evaluation score.
    """
    if metric == 'silhouette_score':
        score = silhouette_score(data, model.labels_)
    elif metric == 'ari':
        score = adjusted_rand_score(labels, model.labels_)
    elif metric == 'nmi':
        score = adjusted_mutual_info_score(labels, model.labels_)
    elif metric == 'fmi':
        score = fowlkes_mallows_score(labels, model.labels_)
    elif metric == 'jaccard':
        score = jaccard_score(labels, model.labels_, average='weighted')
    else:
        raise ValueError(f"Unsupported evaluation metric: {metric}")

    return score


def evaluate_all_models(X, ground_truth_labels, clustering_methods, evaluation_metrics, model_name):
    # Create a DataFrame to store the metric scores
    metric_scores_df = pd.DataFrame(columns=['method', 'metric', 'score'])

    for method, kwargs in clustering_methods.items():
        model = train_model(X, method, **kwargs)

        for metric in evaluation_metrics:
            score = evaluate_model(model, X, ground_truth_labels, metric)
            print(f'{model_name} - {method.capitalize()} {metric.capitalize()}: {score}')

            # Append the metric score to the metric_scores_df DataFrame
            metric_scores_df.loc[len(metric_scores_df)] = [method, metric, score]

    return metric_scores_df


def train_and_evaluate_supervised_model(model, X_train, X_test, y_train, y_test):
    """
    Train and evaluate the given supervised model on the provided data.

    Args:
        model: A scikit-learn model object (e.g., XGBoost or RandomForestClassifier)
        X_train: The training set features
        X_test: The testing set features
        y_train: The training set labels
        y_test: The testing set labels

    Returns:
        cm: The confusion matrix for the model's predictions on the test set
    """

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Make predictions using the trained model on the test set
    y_pred = model.predict(X_test)

    # Calculate the confusion matrix comparing the predictions to the true test set labels
    cm = confusion_matrix(y_test, y_pred)

    # Calculate additional metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"Accuracy: {acc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")

    return cm


def draw_confusion_matrix(cm, labels, title='Confusion Matrix'):
    """Draws a confusion matrix."""
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


def main():
    intermediate_results_path = 'intermediate_results.pk1'
    metric_scores_path = 'metric_scores.pk1'

    if not os.path.exists(intermediate_results_path):
        # Read and merge data
        data1 = read_data("data/16557_pairedTCR_residency_add_index.csv")
        data2 = read_data("data/dt.gene.TCRB.filter.v2.csv")
        data2['barcode_sample_id'] = data2['barcode_sample_id'].str.split('_').str[0]
        merged_data = merge_data(data1, data2, 'cell.barcode', 'barcode_sample_id')

        # Generate k-mers
        kmers_config = {
            'cdr3_column': 'betaClonalSeq',
            'non_overlapping_k': 3,
            'overlapping_k': 3,
            'use_non-overlapping': True,
            'use_overlapping': False
        }
        kmers_data = generate_kmers(merged_data, kmers_config)

        # Create the corpus
        corpus = []
        corpus = create_corpus(kmers_data, 'non_overlapping_sequences', corpus)

        # Train the Doc2Vec model, create vectors, and update the DataFrame
        df = train_and_save_vectors(corpus, kmers_data, 'TCRB_ngram_vectors.csv')

        # Save the intermediate results
        save_pickle({'df': df, 'corpus': corpus}, intermediate_results_path)

    else:
        # Load the intermediate results
        intermediate_results = load_pickle(intermediate_results_path)
        df = intermediate_results['df']
        corpus = intermediate_results['corpus']


    # DATA CLEANING #

    # Clearing nan Celltypes
    df = df.dropna(subset=['celltype'])

    # Convert 'patient.type' column to numerical labels
    cc_ground_truth_labels = df['patient.type'].map({'control': 0, 'case': 1}).values

    # Convert 'celltype' column to numerical labels
    ct_ground_truth_labels = df['celltype'].map(
        {'Memory CD8': 0, 'LTB+ CD4': 1, 'Naive CD8': 2, 'Unk CD8': 3, 'other': 4}).values


    # CLUSTERING METHODS #

    clustering_methods = {
        'kmeans': {'n_clusters': 2},
        'hierarchical': {'n_clusters': 2},
        'dbscan': {'eps': 0.7, 'min_samples': 2},
        'gmm': {'n_components': 2}
    }

    evaluation_metrics = ['silhouette_score', 'ari', 'nmi', 'fmi', 'jaccard']

    # Initialize DataFrame
    metric_scores_df = pd.DataFrame(columns=['Model', 'Method', 'Metric', 'Score'])

    # Define Columns
    af_columns = [col for col in df.columns if col.startswith('X')]
    selected_columns = df.columns[66:2063]

    # Define your models
    X_af_only = df[af_columns].values
    X_af_ge = df[af_columns + selected_columns.tolist()].values
    pca_ge = PCA(n_components=20).fit_transform(df[selected_columns.tolist()].values)
    X_af_pca_ge = np.hstack((df[af_columns].values, pca_ge))
    vector_columns = [col for col in df.columns if col.startswith('vector_')]
    X_kmer_only = df[vector_columns].values
    pca_af = PCA(n_components=20).fit_transform(df[af_columns].values)
    X_pca_af_pca_ge = np.hstack((pca_af, pca_ge))
    X_kmer_af = df[vector_columns + af_columns + selected_columns.tolist()].values
    X_kmer_pca_ge = np.hstack((df[vector_columns].values, pca_ge))
    pca_kmer = PCA(n_components=20).fit_transform(df[vector_columns].values)
    X_pca_kmer_pca_ge = np.hstack((pca_kmer, pca_ge))
    X_kmer_af_ge = df[vector_columns + af_columns + selected_columns.tolist()].values
    X_kmer_af_pca_ge = np.hstack((df[vector_columns + af_columns].values, pca_ge))
    pca_kmer_af = PCA(n_components=20).fit_transform(df[vector_columns + af_columns].values)
    X_pca_kmer_af_pca_ge = np.hstack((pca_kmer_af, pca_ge))

    models = [
        (X_af_only, "Atchley Factors Only Model"),
        (X_af_ge, "Atchley Factors + Genetic Factors Model"),
        (X_af_pca_ge, "Atchley Factors + PCA(Genetic Factors) Model"),
        (X_pca_af_pca_ge, "PCA on Atchley + PCA on GE"),
        (X_kmer_only, "K-mer Exclusive Model"),
        (X_kmer_af, "K-mer & Atchley Factors Model"),
        (X_kmer_pca_ge, "K-mer + PCA(Genetic Factors) Model"),
        (X_pca_kmer_pca_ge, "PCA(K-mer) + PCA(Genetic Factors) Model"),
        (X_kmer_af_ge, "K-mer & Atchley + Genetic Factors Model"),
        (X_kmer_af_pca_ge, "K-mer & Atchley + PCA(Genetic Factors) Model"),
        (X_pca_kmer_af_pca_ge, "PCA(K-mer & Atchley) + PCA(Genetic Factors) Model")
    ]

    if not os.path.exists(metric_scores_path):
        # Run and evaluate all models
        for X, model_name in models:
            for method_name, method_params in clustering_methods.items():
                # train the model using your function
                model = train_model(X, method_name, **method_params)

                for metric in evaluation_metrics:
                    # evaluate the model using your function
                    score = evaluate_model(model, X, cc_ground_truth_labels, metric)
                    print(f'{model_name} - {method_name.capitalize()} {metric.capitalize()}: {score}')

                    # Add result to DataFrame
                    new_row = pd.DataFrame({
                        'Model': [model_name],
                        'Method': [method_name],
                        'Metric': [metric],
                        'Score': [score],
                    })

                    metric_scores_df = pd.concat([metric_scores_df, new_row], ignore_index=True)

        # Save the metric scores to a new CSV file
        metric_scores_df.to_csv('clustering_metric_scores.csv', index=False)
        save_pickle({'metric_scores_df': metric_scores_df}, metric_scores_path)

    else:
        # Load the metric scores
        metric_scores_data = load_pickle(metric_scores_path)
        metric_scores_df = metric_scores_data['metric_scores_df']


    # SUPERVISED METHODS #

    df_pca_combined = pd.DataFrame(X_pca_kmer_af_pca_ge)
    df_reg_combined = pd.DataFrame(X_kmer_af_ge)

    # Patient Case/Control Classifiers

    # Split the case/control data into training and testing sets (K-mer, Atchley, and Genetic Factors))
    X_cctrain, X_cctest, y_cctrain, y_cctest = train_test_split(df_reg_combined, cc_ground_truth_labels,
                                                                test_size=0.2, random_state=42)

    # Split the case/control data into training and testing sets (PCA(K-mer & Atchley) + PCA(Genetic Factors))
    X_pca_cctrain, X_pca_cctest, y_pca_cctrain, y_pca_cctest = train_test_split(df_pca_combined, cc_ground_truth_labels,
                                                                                test_size=0.2, random_state=42)

    # Train a RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    print("Random Forest Classifier : Case/Control : Regular Data")
    rf_cc_cm = train_and_evaluate_supervised_model(rf, X_cctrain, X_cctest, y_cctrain, y_cctest)
    print("Random Forest Classifier : Case/Control : PCA Data")
    rf_pca_cc_cm = train_and_evaluate_supervised_model(rf, X_pca_cctrain, X_pca_cctest, y_pca_cctrain, y_pca_cctest)

    # Train a XGBoost Classifier
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    print("XGB Classifier : Case/Control : Regular Data")
    xgb_cc_cm = train_and_evaluate_supervised_model(xgb, X_cctrain, X_cctest, y_cctrain, y_cctest)
    print("XGB Classifier : Case/Control : PCA Data")
    xgb_pca_cc_cm = train_and_evaluate_supervised_model(xgb, X_pca_cctrain, X_pca_cctest, y_pca_cctrain, y_pca_cctest)

    # Patient Celltype Classifier

    # Split the case/control data into training and testing sets (K-mer, Atchley, and Genetic Factors))
    X_cttrain, X_cttest, y_cttrain, y_cttest = train_test_split(df_reg_combined, ct_ground_truth_labels, test_size=0.2,
                                                                random_state=42)

    # Split the case/control data into training and testing sets (PCA(K-mer & Atchley) + PCA(Genetic Factors))
    X_pca_cttrain, X_pca_cttest, y_pca_cttrain, y_pca_cttest = train_test_split(df_pca_combined, ct_ground_truth_labels,
                                                                                test_size=0.2, random_state=42)

    # Train a RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    print("Random Forest Classifier : CellType : Regular Data")
    rf_ct_cm = train_and_evaluate_supervised_model(rf, X_cttrain, X_cttest, y_cttrain, y_cttest)
    print("Random Forest Classifier : CellType : PCA Data")
    rf_pca_ct_cm = train_and_evaluate_supervised_model(rf, X_pca_cttrain, X_pca_cttest, y_pca_cttrain, y_pca_cttest)

    # Train a XGBoost Classifier
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    print("XGB Classifier : CellType : Regular Data")
    xgb_ct_cm = train_and_evaluate_supervised_model(xgb, X_cttrain, X_cttest, y_cttrain, y_cttest)
    print("XGB Classifier : CellType : PCA Data")
    xgb_pca_ct_cm = train_and_evaluate_supervised_model(xgb, X_pca_cttrain, X_pca_cttest, y_pca_cttrain, y_pca_cttest)


    # VISUALIZATION #

    # Cluster Method Performance
    heatmap_data = metric_scores_df.pivot_table(index='Method', columns='Metric', values='Score')

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, cmap='viridis')
    plt.title('Clustering methods performance')
    plt.show()

    # Cluster Model Performance
    heatmap_data = metric_scores_df.pivot_table(index='Model', columns='Metric', values='Score')

    fig, ax = plt.subplots(figsize=(12, 8))  # create a figure and a set of subplots
    sns.heatmap(heatmap_data, annot=True, cmap='viridis', ax=ax)  # plot your data
    plt.title('Clustering Model performance')
    plt.subplots_adjust(bottom=0.15, left=0.30)  # adjust the bottom and left margins
    plt.show()

    # For case/control classifiers
    draw_confusion_matrix(rf_cc_cm, ['control', 'case'], 'Random Forest Confusion Matrix for Case/Control')
    draw_confusion_matrix(rf_pca_cc_cm, ['control', 'case'], 'Random Forest Confusion Matrix for PCA-Case/Control')
    draw_confusion_matrix(xgb_cc_cm, ['control', 'case'], 'XGBoost Confusion Matrix for Case/Control')
    draw_confusion_matrix(xgb_pca_cc_cm, ['control', 'case'], 'XGBoost Confusion Matrix for PCA-Case/Control')

    # For celltype classifiers
    draw_confusion_matrix(rf_ct_cm, ['Memory CD8', 'LTB+ CD4', 'Naive CD8', 'Unk CD8', 'other'],
                          'Random Forest Confusion Matrix for Celltype')
    draw_confusion_matrix(rf_pca_ct_cm, ['Memory CD8', 'LTB+ CD4', 'Naive CD8', 'Unk CD8', 'other'],
                          'Random Forest Confusion Matrix for PCA-Celltype')
    draw_confusion_matrix(xgb_ct_cm, ['Memory CD8', 'LTB+ CD4', 'Naive CD8', 'Unk CD8', 'other'],
                          'XGBoost Confusion Matrix for Celltype')
    draw_confusion_matrix(xgb_pca_ct_cm, ['Memory CD8', 'LTB+ CD4', 'Naive CD8', 'Unk CD8', 'other'],
                          'XGBoost Confusion Matrix for PCA-Celltype')


if __name__ == "__main__":
    main()
