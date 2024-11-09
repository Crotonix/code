import numpy as np
from decisionTree import DecisionTree


class RandomForestClassfier:
    """
    Random Forest Classfier
    Training: User "train" function with train set features and labels
    Predicting: Use "predict" fucntion with test set features"""

    def __init__(
        self,
        n_base_learner=10,
        max_depth=4,
        min_samples_leaf=1,
        min_information_gain=0.0,
        numb_of_features_splitting=None,
        bootstrap_sample_size=None,
    ) -> None:
        self.n_base_learner = n_base_learner
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.numb_of_features_splitting = numb_of_features_splitting
        self.bootstrap_sample_size = bootstrap_sample_size

    def _create_bootstrap_samples(self, x, y) -> tuple:
        """Create bootstrap samples for each base learner, Bootstrapping is sampling with replcement with n_base_learner subsets with bootstrap_sample_size as size"""
        bootstrap_samples_x = []
        bootstrap_samples_y = []

        for _ in range(self.n_base_learner):
            if not self.bootstrap_sample_size:
                self.bootstrap_sample_size = x.shape[0]

            sampled_idx = np.random.choice(
                x.shape[0], size=self.bootstrap_sample_size, replace=True
            )
            bootstrap_samples_x.append(x[sampled_idx])
            bootstrap_samples_y.append(y[sampled_idx])

        return bootstrap_samples_x, bootstrap_samples_y

    def train(self, x_train: np.array, y_train: np.array) -> None:
        """Train the model with the given x and y datasets"""

        bootstrap_samples_x, bootstrap_samples_y = self._create_bootstrap_samples(
            x_train, y_train
        )

        self.base_learner_list = []
        for base_learner_idx in range(self.n_base_learner):
            base_learner = DecisionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_infromation_gain=self.min_information_gain,
                numb_of_features_splitting=self.numb_of_features_splitting,
            )

            base_learner.train(
                bootstrap_samples_x[base_learner_idx],
                bootstrap_samples_y[base_learner_idx],
            )
            self.base_learner_list.append(base_learner)

        # self.feature_importances = self._calculate_feature_importances(
        #     self.base_learner_list
        # )

    def _predict_proba_w_base_learners(self, x_set: np.array) -> list:
        """Creates list of predictions for all base learners"""
        pred_prob_list = []
        for base_learner in self.base_learner_list:
            pred_prob_list.append(base_learner.predict_proba(x_set))

        return pred_prob_list

    def predict_proba(self, x_set: np.array) -> list:
        """Returns the predicted probs for a given data set"""

        pred_probs = []
        base_learner_pred_probs = self._predict_proba_w_base_learners(x_set)

        # Average the predicted proababilties for base learners
        for obs in range(x_set.shape[0]):
            base_learner_probs_for_obs = [a[obs] for a in base_learner_pred_probs]
            # Calculate the average for each index
            obs_average_pred_probs = np.mean(base_learner_probs_for_obs, axis=1)
            pred_probs.append(obs_average_pred_probs)

        return pred_probs

    def predict(self, x_set: np.array) -> np.array:
        """Returns the predicted labels for a given data set"""

        pred_probs = self.predict_proba(x_set)
        preds = np.argmax(pred_probs, axis=1)

        return preds
