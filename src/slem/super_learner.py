# flake8: noqa: E741
#
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from scipy.optimize import minimize
from scipy.optimize import nnls
from sklearn.preprocessing import PolynomialFeatures  #
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import (
    ElasticNet,
    LinearRegression,
    LogisticRegression,
    BayesianRidge,
)
from sklearn.svm import SVR, SVC
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    AdaBoostClassifier,
    RandomForestClassifier,
)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from warnings import filterwarnings
from sklearn.naive_bayes import GaussianNB
from typing import Optional, Union, Any, Dict, List
from numpy.typing import NDArray


def fn(X, A, b):
    return np.linalg.norm(A.dot(X) - b)


""" An example dictionary of estimators can be specified as follows:
Gest_dict = {'LR': LogisticRegression(), 'SVC': SVC(probability=True),
                                 'RF': RandomForestClassifier(), 'KNN': KNeighborsClassifier(),
                                 'AB': AdaBoostClassifier(), 'poly': 'poly'}

Note that 'poly' is specified as a string because there are no default polynomial feature regressors in sklearn.
This one defaults to 2nd order features (e.g. x1*x2, x1*x3 etc...)"""


def combiner_solve(x, y):
    # adapted from https://stackoverflow.com/questions/33385898/how-to-include-constraint-to-scipy-nnls-function-solution-so-that-it-sums-to-1/33388181
    beta_0, rnorm = nnls(x, y)
    cons = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds = [[0.0, None]] * x.shape[1]
    minout = minimize(
        fn, beta_0, args=(x, y), method="SLSQP", bounds=bounds, constraints=cons
    )
    beta = minout.x
    return beta


class SuperLearner(object):
    def __init__(
        self,
        output: str,
        k: int,
        standardized_outcome: bool = False,
        calibration: bool = True,
        learner_list: Optional[List[str]] = None,
    ):
        if learner_list is None:
            self.learner_list = ["Elastic", "LR", "MLP", "SV", "AB", "RF", "BR", "poly"]
        else:
            self.learner_list = learner_list

        self.orig_learner_list = self.learner_list

        self.num_learners = len(self.learner_list)
        self.k = k  # number of cross validation folds
        self.output = output  # 'reg' for regression, 'proba' or 'cls' classification
        self.trained_superlearner = None
        self.est_dict: Dict[str, Any]  # dictionary of learners/algos
        self.standardized_outcome = standardized_outcome
        self.calibration = calibration

        self.X_std: float
        self.X_mean: float
        self.y_std: float
        self.y_mean: float
        self.num_classes: int
        self._init_learners()

    def _init_learners(self):
        est_dict = {}
        for learner in self.orig_learner_list:
            if learner == "Elastic":
                l = "Elastic0.25"
                est_dict[l] = ElasticNet(alpha=0.25)
                l = "Elastic0.5"
                est_dict[l] = ElasticNet(alpha=0.5)
                l = "Elastic0.75"
                est_dict[l] = ElasticNet(alpha=0.75)
                l = "Elastic1"
                est_dict[l] = ElasticNet(alpha=0.1)
            elif learner == "LR":
                if (
                    (self.output == "cls")
                    or (self.output == "proba")
                    or (self.output == "cat")
                ):
                    est_dict[learner] = LogisticRegression(max_iter=500)
                else:
                    est_dict[learner] = LinearRegression()
            elif learner == "MLP":
                if (
                    (self.output == "cls")
                    or (self.output == "proba")
                    or (self.output == "cat")
                ):
                    l = "MLP-1"
                    est_dict[l] = MLPClassifier(
                        hidden_layer_sizes=(
                            100,
                            100,
                            100,
                        ),
                        alpha=0.001,
                        max_iter=5000,
                    )
                    l = "MLP-2"
                    est_dict[l] = MLPClassifier(
                        hidden_layer_sizes=(
                            50,
                            50,
                            50,
                        ),
                        alpha=0.001,
                        max_iter=4000,
                    )
                    l = "MLP-3"
                    est_dict[l] = MLPClassifier(
                        hidden_layer_sizes=(
                            50,
                            50,
                        ),
                        alpha=0.001,
                        max_iter=3000,
                    )
                    l = "MLP-4"
                    est_dict[l] = MLPClassifier(
                        hidden_layer_sizes=(
                            30,
                            30,
                        ),
                        alpha=0.001,
                        max_iter=2000,
                    )
                else:
                    l = "MLP-1"
                    est_dict[l] = MLPRegressor(
                        hidden_layer_sizes=(
                            100,
                            100,
                            100,
                        ),
                        alpha=0.001,
                        max_iter=5000,
                    )
                    l = "MLP-2"
                    est_dict[l] = MLPRegressor(
                        hidden_layer_sizes=(
                            50,
                            50,
                            50,
                        ),
                        alpha=0.001,
                        max_iter=4000,
                    )
                    l = "MLP-3"
                    est_dict[l] = MLPRegressor(
                        hidden_layer_sizes=(
                            50,
                            50,
                        ),
                        alpha=0.001,
                        max_iter=3000,
                    )
                    l = "MLP-4"
                    est_dict[l] = MLPRegressor(
                        hidden_layer_sizes=(
                            30,
                            30,
                        ),
                        alpha=0.001,
                        max_iter=2000,
                    )
            elif learner == "SV":
                if (
                    (self.output == "cls")
                    or (self.output == "proba")
                    or (self.output == "cat")
                ):
                    est_dict[learner] = SVC(probability=True)
                else:
                    est_dict[learner] = SVR()
            elif learner == "AB":
                if (
                    (self.output == "cls")
                    or (self.output == "proba")
                    or (self.output == "cat")
                ):
                    l = "AB-1"
                    est_dict[l] = AdaBoostClassifier(n_estimators=100)
                    l = "AB-2"
                    est_dict[l] = AdaBoostClassifier(n_estimators=50)
                    l = "AB-3"
                    est_dict[l] = AdaBoostClassifier(n_estimators=25)
                    l = "AB-4"
                    est_dict[l] = AdaBoostClassifier(n_estimators=10)
                else:
                    l = "AB-1"
                    est_dict[l] = AdaBoostRegressor(n_estimators=100)
                    l = "AB-2"
                    est_dict[l] = AdaBoostRegressor(n_estimators=50)
                    l = "AB-3"
                    est_dict[l] = AdaBoostRegressor(n_estimators=25)
                    l = "AB-4"
                    est_dict[l] = AdaBoostRegressor(n_estimators=10)

            elif learner == "RF":
                if (
                    (self.output == "cls")
                    or (self.output == "proba")
                    or (self.output == "cat")
                ):
                    l = "RF-1"
                    est_dict[l] = RandomForestClassifier(n_estimators=200)
                    l = "RF-2"
                    est_dict[l] = RandomForestClassifier(n_estimators=100)
                    l = "RF-3"
                    est_dict[l] = RandomForestClassifier(n_estimators=50)
                    l = "RF-4"
                    est_dict[l] = RandomForestClassifier(n_estimators=20)
                else:
                    l = "RF-1"
                    est_dict[l] = RandomForestRegressor(n_estimators=300)
                    l = "RF-2"
                    est_dict[l] = RandomForestRegressor(n_estimators=100)
                    l = "RF-3"
                    est_dict[l] = RandomForestRegressor(n_estimators=50)
                    l = "RF-4"
                    est_dict[l] = RandomForestRegressor(n_estimators=20)

            elif (learner == "BR") or (learner == "NB"):
                if (
                    (self.output == "cls")
                    or (self.output == "proba")
                    or (self.output == "cat")
                ):
                    est_dict[learner] = GaussianNB()
                else:
                    est_dict[learner] = BayesianRidge()

            elif learner == "poly":
                est_dict[learner] = "poly"

        self.est_dict = est_dict
        self.num_learners = len(list(self.est_dict.keys()))
        self.learner_list = list(est_dict.keys())

    def fit(
        self, X: Union[pd.DataFrame, NDArray[Any]], y: Union[pd.DataFrame, NDArray[Any]]
    ):
        filterwarnings("ignore")
        X = X.values.astype("float") if isinstance(X, pd.DataFrame) else X
        y = y.values[:, 0].astype("float") if isinstance(y, pd.DataFrame) else y

        X = X.values.astype("float") if isinstance(X, pd.Series) else X
        y = y.values.astype("float") if isinstance(y, pd.Series) else y

        # mean and std for full dataset (can be reused wth new data at prediction time)
        self.X_std = X.std(0)
        self.X_mean = X.mean(0)

        if self.standardized_outcome:
            self.y_std = y.std(0)
            self.y_mean = y.mean(0)

        if (self.output == "cls") or (self.output == "proba"):
            self.num_classes = len(np.unique(y))

            if self.num_classes > 2:
                self.output = "cat"
        else:
            self.num_classes = 1

        if (self.output == "cls") or (self.output == "proba") or (self.output == "cat"):
            kf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=0)
        else:
            kf = KFold(n_splits=self.k, shuffle=True, random_state=0)

        self._init_learners()

        all_preds = np.zeros((len(y), self.num_learners))  # for test preds

        i = 0
        for key in self.learner_list:
            preds = []
            gts = []

            for train_index, test_index in kf.split(X, y):
                self._init_learners()  # initialise fresh learners

                est = self.est_dict[key]
                X_train = X[train_index]
                X_test = X[test_index]
                y_train = y[train_index]
                y_test = y[test_index]

                # per train/test fold means and standard deviations
                X_std = X_train.std(0)
                X_mean = X_train.mean(0)
                X_train = (X_train - X_mean) / X_std
                X_test = (X_test - X_mean) / X_std

                if self.standardized_outcome:
                    y_std = y_train.std(0)
                    y_mean = y_train.mean(0)
                    y_train = (y_train - y_mean) / y_std
                    y_test = (y_test - y_mean) / y_std

                if key == "poly":
                    est = (
                        LogisticRegression(C=1e2, max_iter=350)
                        if (
                            (self.output == "cls")
                            or (self.output == "proba")
                            or (self.output == "cat")
                        )
                        else LinearRegression()
                    )
                    poly = PolynomialFeatures(2)
                    X_train_poly = poly.fit_transform(X_train)
                    X_test_poly = poly.fit_transform(X_test)
                    if (
                        (self.output == "cls")
                        or (self.output == "proba")
                        or (self.output == "cat")
                    ) and self.calibration:
                        est = CalibratedClassifierCV(base_estimator=est, cv=5)
                    est.fit(X_train_poly, y_train)

                else:
                    if (
                        (self.output == "cls")
                        or (self.output == "proba")
                        or (self.output == "cat")
                    ) and self.calibration:
                        est = CalibratedClassifierCV(base_estimator=est, cv=5)

                    est.fit(X_train, y_train)

                p = est.predict(X_test_poly) if key == "poly" else est.predict(X_test)
                preds.append(p)
                gts.append(y_test)

            preds = np.concatenate(preds)
            gts = np.concatenate(gts)

            all_preds[:, i] = preds

            i += 1

        # estimate betas on test predictions
        self.beta = combiner_solve(
            all_preds, gts
        )  # all_preds is of shape [batch, categories, predictors]

        # now train each estimator on full dataset

        self._init_learners()  # initialise learners fresh

        X = (X - self.X_mean) / self.X_std
        if self.standardized_outcome:
            y = (y - self.y_mean) / self.y_std

        for key in self.est_dict.keys():
            est = self.est_dict[key]

            if key == "poly":
                est = (
                    LogisticRegression(C=1e2, max_iter=350)
                    if (
                        (self.output == "cls")
                        or (self.output == "proba")
                        or (self.output == "cat")
                    )
                    else LinearRegression()
                )
                poly = PolynomialFeatures(2)
                X_poly = poly.fit_transform(X)

                est.fit(X_poly, y)

            else:
                est.fit(X, y)

            self.est_dict[key] = est

        return (
            all_preds,
            gts,
        )  # returns the test folds from k-fold process along with ground truth

    def predict(self, X: Union[pd.DataFrame, NDArray[Any]]):
        X = X.values.astype("float") if isinstance(X, pd.DataFrame) else X
        X = X.values.astype("float") if isinstance(X, pd.Series) else X
        X_ = (X - self.X_mean) / self.X_std
        all_preds = np.zeros((len(X_), self.num_learners))
        i = 0

        for key in self.est_dict.keys():
            est = self.est_dict[key]
            if key == "poly":
                poly = PolynomialFeatures(2)
                X_scaled = poly.fit_transform(X_)

            preds = est.predict(X_) if key != "poly" else est.predict(X_scaled)

            all_preds[:, i] = preds

            i += 1

        weighted_preds = np.dot(all_preds, self.beta)
        weighted_preds = weighted_preds.reshape(-1, 1)
        if self.standardized_outcome:
            weighted_preds = (weighted_preds * self.y_std) + self.y_mean
        return weighted_preds

    def predict_proba(self, X: Union[pd.DataFrame, NDArray[Any]]):
        X = X.values.astype("float") if isinstance(X, pd.DataFrame) else X
        X = X.values.astype("float") if isinstance(X, pd.Series) else X
        X_ = (X - self.X_mean) / self.X_std

        all_preds = np.zeros((len(X), self.num_classes, self.num_learners))
        i = 0

        for key in self.est_dict.keys():
            est = self.est_dict[key]
            if key == "poly":
                poly = PolynomialFeatures(2)
                X_poly = poly.fit_transform(X_)

            preds = (
                est.predict_proba(X_) if key != "poly" else est.predict_proba(X_poly)
            )
            all_preds[:, :, i] = preds

            i += 1

        weighted_preds_list = []
        for cl in range(self.num_classes):
            preds = np.dot(all_preds[:, cl, :], self.beta)

            if self.standardized_outcome:
                preds = (preds * self.y_std) + self.y_mean

            weighted_preds_list.append(preds)

        weighted_preds = np.asarray(weighted_preds_list).T

        return weighted_preds
