import numpy as np
from treenode import TreeNode
from collections import Counter


class DecisionTree:
    """
    Decision Tree classifier
    Training: User "train" function with train set features and labels
    Predicting: User "predict" function with test set features
    """

    def __init__(
        self,
        max_depth=4,
        min_samples_leaf=1,
        min_infromation_gain=0.0,
        numb_of_features_splitting=None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_infromation_gain
        self.number_of_features_splitting = numb_of_features_splitting

    def _entropy(self, class_probabilities: list) -> float:
        # first get correct domain for log function as p can be 0
        # probs = [p for p in class_probabilities if p > 0]
        # for vectors numpy is faster
        # return np.sum(np.log2(probs) * np.array(probs) * -1)
        return sum([-p * np.log2(p) for p in class_probabilities if p > 0])

    def _class_probabilties(self, labels: list) -> list:
        total_count = len(labels)
        # return {
        #     label: (label_count / total_count)
        #     for label, label_count in Counter(labels).items()
        # }
        return [label_count / total_count for label_count in Counter(labels).values()]

    def _data_entropy(self, labels: int) -> float:
        # return self._entropy(self._class_probabilties(labels).values())
        return self._entropy(self._class_probabilties(labels))

    def _partition_entropy(self, subsets: list) -> float:
        """subsets = list of label lists [[1, 0, 0], [1, 1, 1]]"""
        total_count = sum([len(subset) for subset in subsets])
        return sum(
            [
                self._data_entropy(subset) * (len(subset) / total_count)
                for subset in subsets
            ]
        )

    def _split(self, data: np.array, feature_idx: int, feature_val: float) -> tuple:
        mask_below_threshold = data[:, feature_idx] < feature_val
        group1 = data[mask_below_threshold]
        group2 = data[~mask_below_threshold]

        return group1, group2

    def _select_features_to_use(self, data: np.array) -> list:
        """
        Randomly selects features to use while splitting w.r.t. hyperparameter numb_of_features_splitting
        """
        feature_idx = list(range(data.shape[1] - 1))

        if self.numb_of_features_splitting == "sqrt":
            feature_idx_to_use = np.random.choice(
                feature_idx, size=int(np.sqrt(len(feature_idx)))
            )
        elif self.numb_of_features_splitting == "log":
            feature_idx_to_use = np.random.choice(
                feature_idx, size=int(np.log2(len(feature_idx)))
            )
        else:
            feature_idx_to_use = feature_idx

        return feature_idx_to_use

    def find_best_split(self, data: np.array) -> tuple:
        """
        Find the bes split(with lowest entropy meaning no dominant class) given data
        Returns 2 splitted group
        """
        min_part_entropy = 1e9
        min_part_entropy_feature_idx = None
        min_entropy_feature_val = None
        g1_min, g2_min = None, None

        for idx in range(data.shape[1] - 1):
            feature_val = np.median(data[:, idx])
            g1, g2 = self._split(data, idx, feature_val)
            part_entropy = self._partition_entropy([g1[:, -1], g2[:, -1]])
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

    def _find_label_probs(self, data: np.array) -> np.array:
        labels_as_integers = data[:, -1].astype(int)
        # Calculate the number of labels
        total_labels = len(labels_as_integers)
        # Calculate the ratios (probabilities) for each label
        label_probabilities = np.zeros(len(self.labels_in_train), dtype=float)

        # Populate the label_probabilties array based on the specific labels
        for i, label in enumerate(self.labels_in_train):
            label_index = np.where(labels_as_integers == i)[0]
            if len(label_index) > 0:
                label_probabilities[i] = len(label_index) / total_labels

        return label_probabilities

    def _create_tree(self, data: np.array, current_depth: int) -> TreeNode:
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
        label_probabilities = self._find_label_probs(data)

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
        node.left = self._create_tree(split_1_data, current_depth)
        node.right = self._create_tree(split_2_data, current_depth)

        return node

    def train(self, x_train: np.array, y_train: np.array) -> None:
        # Concat features and labels
        self.labels_in_train = np.unique(y_train)
        train_data = np.concatenate((x_train, np.reshape(y_train, (-1, 1))), axis=1)

        # start creating the trees
        self.tree = self._create_tree(data=train_data, current_depth=0)

        # Calculate feature importance
        # self.feature_importances = dict.fromkeys(range(x_train.shape[1]), 0)
        # self._calculate_feature_importance(self.tree)
        # # Noramalize the feature importance values
        # self.feature_importances = {
        #     k: v / total
        #     for total in (sum(self.feature_importances.values()),)
        #     for k, v in self.feature_importances.items()
        # }

    def predict_one_sample(self, x: np.array) -> np.array:
        """Returns prediction for 1 dim array"""
        node = self.tree

        # Find the leaf which x belongs
        while node:
            pred_probs = node.prediction_probs
            if x[node.feature_idx] < node.feature_val:
                node = node.left
            else:
                node = node.right
        return pred_probs

    def predict_proba(self, x_set: np.array) -> np.array:
        """Returns the predicted probs for a given data set"""

        pred_probs = np.apply_along_axis(self.predict_one_sample, 1, x_set)
        return pred_probs

    def predict(self, x_set: np.array) -> np.array:
        """Returns the predicted probs for a given data set"""
        pred_probs = self.predict_proba(x_set)
        preds = np.argmax(pred_probs, axis=1)

        return preds

    def _print_recursive(self, node: TreeNode, level=0) -> None:
        if node != None:
            self._print_recursive(node.left, level + 1)
            print(" " * 4 * level + "->" + node.node_def())
            self._print_recursive(node.right, level + 1)

    def print_tree(self) -> None:
        self._print_recursive(node=self.tree)

    # def _calculate_feature_importance(self, node):
    #     """Calculates the feature importance by visiting each node in the tree recursively"""
    #     if node != None:
    #         self.feature_importances[node.feature_idx] += node.feature_importance
    #         self._calculate_feature_importance(node.left)
    #         self._calculate_feature_importance(node.right)
