# Mastering-Decision-Trees-A-Hands-On-Approach
Decision trees are powerful tools in the realm of machine learning, offering transparent and interpretable solutions to classification and regression problems. In this practical lab, we'll delve into the world of decision trees by implementing one from scratch and applying it to the critical task of classifying whether a mushroom is edible or poisonous. So let's roll up our sleeves and dive in!

1. Packages
Before we embark on our journey, let's ensure we have the necessary tools at our disposal. We'll primarily be using Python and a few of its libraries such as NumPy for data manipulation.
pythonCopy code
import numpy as np  

2. Problem Statement
Our task is to develop a decision tree classifier that can accurately determine whether a mushroom is edible or poisonous based on certain features. This is a classic binary classification problem, where each mushroom in our dataset will be labeled as either 'edible' or 'poisonous'.

3. Dataset
The dataset we'll be working with contains various attributes of mushrooms along with their corresponding class labels ('edible' or 'poisonous'). 

4.1 Calculate Entropy
Entropy is a measure of impurity or disorder in a dataset. For a binary classification problem, entropy is calculated as:
where �1p1 and �2p2 are the proportions of positive and negative instances in the dataset �S, respectively.
Exercise 1
Write a function to calculate entropy given the proportions of positive and negative instances.
def compute_entropy(y):
    entropy = 0.
    n_edible = np.count_nonzero(y)
    n_poisonous = len(y) - n_edible
    # Check for no variation in the labels
    if n_edible == 0 or n_poisonous == 0:
        return 0
    # Compute the probability of each label
    p_edible = n_edible / len(y)
    p_poisonous = n_poisonous / len(y)
    # Entropy formula
    entropy = -(p_edible * np.log2(p_edible) + p_poisonous * np.log2(p_poisonous))
    return entropy

4.2 Split Dataset
Splitting the dataset involves dividing it into subsets based on the values of a specific attribute. This process aims to maximize the homogeneity of the resulting subsets.
Exercise 2
Write a function to split the dataset based on a given attribute and its value.
python

def split_dataset(X, node_indices, feature)
    left_indices = []
    right_indices = []
    for idx in node_indices:
        # If the feature value is 1, add the index to left_indices
        if X[idx, feature] == 1:
            left_indices.append(idx)
        # Else, add the index to right_indices
        else:
            right_indices.append(idx)
    return left_indices, right_indices

4.3 Calculate Information Gain
Information gain measures the effectiveness of a particular attribute in classifying the data. It is calculated as the difference between the entropy of the parent dataset and the weighted sum of entropies of its child subsets.
Exercise 3
Write a function to calculate information gain given a dataset and its subsets.
def compute_information_gain(X, y, node_indices, feature):
    # Split dataset
    left_indices, right_indices = split_dataset(X, node_indices, feature)
    # Some useful variables
    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]
    information_gain = 0
    node_entropy = compute_entropy(y_node)
    left_entropy = compute_entropy(y_left)
    right_entropy = compute_entropy(y_right)
    # Compute the weight of each child node
    left_weight = len(y_left) / len(y_node)
    right_weight = len(y_right) / len(y_node)

    # Compute the information gain
    information_gain = node_entropy - (left_weight * left_entropy + right_weight * right_entropy)    
    return information_gain

4.4 Get Best Split
To determine the best attribute to split on, we calculate the information gain for each attribute and choose the one with the highest gain.
Exercise 4
Write a function to find the best attribute to split on.
def get_best_split(X, y, node_indices):   
    num_features = X.shape[1]
    best_feature = -1
    best_information_gain = -np.inf
    best_threshold = None
    # Iterate over each feature
    for feature_index in range(num_features):
        # Get unique feature values
        unique_values = np.unique(X[node_indices, feature_index])
        # Iterate over potential thresholds
        for threshold in unique_values:
            # Compute information gain for this split
            information_gain = compute_information_gain(X, y, node_indices, feature_index)  
            # Update best split if this one is better
            if information_gain > best_information_gain:
                best_information_gain = information_gain
                best_feature = feature_index
                best_threshold = threshold
    return best_feature

Decision trees offer a transparent and interpretable solution to classification problems, making them invaluable tools in the field of machine learning. Keep exploring and experimenting with different datasets to further hone skills!
