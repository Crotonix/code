import numpy as np


class TreeNode:
    def __init__(
        self, data, feature_idx, feature_val, prediction_prbs, information_gain
    ) -> None:
        self.data = data
        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.prediction_probs = prediction_prbs
        self.information_gain = information_gain
        self.left = None
        self.right = None


class DecisionTree:
    """
    Decision Tree classifier
    Training: User "train" function with train set features and labels
    Predicting: User "predict" function with test set features
    """

    def __init__(
        self, max_depth=4, min_samples_leaf=1, min_infromation_gain=0.0
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_infromation_gain

    def _entropy(self, class_probabilities: list) -> float:
        # first get correct domain for log function
        probs = [p for p in class_probabilities if p > 0]
        # for vectors numpy is faster
        return np.sum(np.log2(probs) * np.array(probs))

    def find_best_split(self, data: np.array) -> tuple:
        """
        Find the bes split(with lowest entropy meaning no dominant class) given data
        Returns 2 splitted group
        """
        min_part_entropy = 1e-6
        min_part_entropy_feature_idx = None
        min_entropy_feature_val = None
        g1_min, g2_min = None, None

        for idx in range(data.shape[1] - 1):
            feature_val = np.median(data[:, idx])
            g1, g2 = self.split(data, idx, feature_val)
            part_entropy = self.partition_entropy([g1[:, -1], g2[:, -1]])
            if part_entropy < min_part_entropy:
                min_part_entropy = part_entropy
                min_part_entropy_feature_idx = idx
                min_entropy_feature_val = feature_val
                g1_min, g2_min = g1, g2
        return (
            g1_min,
            g2_min,
            min_part_entropy_feature_idx,
            min_entropy_feature_val,
            min_part_entropy,
        )

    def create_tree(self, data: np.array, current_depth: int) -> TreeNode:
        # Check if the max depth has been reached (stopping criteria)
        if current_depth >= self.max_depth:
            return None

        # Find best split
        (
            split_1_data,
            split_2_data,
            split_feature_idx,
            split_feature_val,
            split_feature_entropy,
        ) = self.find_best_split(data)

        # Find label probs for the node
        label_probabilities = self.find_label_probs(data)

        # Calculate Information Gain
        node_entropy = self._entropy(label_probabilities)
        information_gain = node_entropy - split_feature_entropy

        # Create node
        node = TreeNode(
            data,
            split_feature_idx,
            split_feature_val,
            label_probabilities,
            information_gain,
        )

        # Check if the min_samples_leaf has been satisfied (stopping criteria)
        if (
            self.min_samples_leaf > split_1_data.shape[0]
            or self.min_samples_leaf > split_2_data.shape[0]
        ):
            return node
        # Check if the min_information_gain has been satisfied (stopping criteria)
        elif information_gain < self.min_information_gain:
            return node

        current_depth += 1
        node.left = self.create_tree(split_1_data, current_depth)
        node.right = self.create_tree(split_2_data, current_depth)

        return node
