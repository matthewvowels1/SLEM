import networkx as nx
import numpy as np
from .super_learner import SuperLearner
from .utils import get_full_ordering, reorder_dag
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    explained_variance_score,
    median_absolute_error,
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import joblib
import os
from pathlib import Path
import tqdm
import pandas as pd
from typing import Optional, Union, List, Dict


class DAGLearner:
    def __init__(
        self,
        dag: nx.DiGraph,
        var_types: Dict[str, str],
        seed: int = 42,
        k: int = 5,
        save_models: bool = False,
        model_dir: Optional[str] = None,
        learner_list: Optional[List[str]] = None,
        baseline: bool = False,
    ):
        """
        DAGLearner builds a set of SuperLearners according to a specified DAG for the purposes of data-driven/non-linear causal effect estimation
        :param dag: networkx DAG DiGraph object
        :param var_types:  a dictionary of variable names with continuous = 'cont', binary = 'bin' or categorical = 'cat types
        :param seed: random seed as an integer
        :param k: the number of folds in the k-fold CV fitting process
        :param save_models: whether to save the trained models
        :param model_dir: where to save the trained models (if save_models is True)
        :param learner_list: a list of superlearner learners, can be 'Elastic', 'LR', 'MLP', 'SV', 'AB', 'RF', 'BR', 'poly'
        """
        # Default parameters, can be overridden by params

        if learner_list is None:
            learner_list = ["Elastic", "LR", "MLP", "SV", "AB", "RF", "BR", "poly"]

        self.learner_list = learner_list
        self.baseline = baseline
        if self.baseline:
            self.learner_list = ["LR"]  # use logistic/linear regression as the baseline
        self.model_dir = model_dir
        self.var_types = var_types
        self.k = k
        self.save_models = save_models
        dag = reorder_dag(dag=dag)  # topologically sort
        self.causal_ordering = get_full_ordering(
            dag
        )  # get causal ordering of each (sorted) variable
        self.dag = dag  # update dag with sorted dag
        self.seed = seed

        self.regression_metrics = {
            "r2": r2_score,
            "mae": mean_absolute_error,
            "mse": mean_squared_error,
            "expl_var": explained_variance_score,
            "medae": median_absolute_error,
        }
        self.classification_metrics = {
            "acc": accuracy_score,
            "balacc": balanced_accuracy_score,
            "prec": precision_score,
            "rec": recall_score,
            "f1": f1_score,
        }

        self.models = {}
        self.predictors = {}
        # for each variable, if it has parents, assign a SuperLearner according to its type and get its parents as predictors
        for i, var in enumerate(self.causal_ordering.keys()):
            order = self.causal_ordering[var]
            var_type = self.var_types[var]
            if order >= 1:
                if var_type == "cont":  # get types for superlearner
                    output_type = "reg"
                elif var_type == "cat" or var_type == "bin":
                    output_type = "cat"

                self.models[var] = SuperLearner(
                    output=output_type,
                    k=self.k,
                    standardized_outcome=False,
                    calibration=False,
                    learner_list=self.learner_list,
                )

                parent_list = list(self.dag.predecessors(var))
                self.predictors[var] = parent_list

    def fit(self, data: pd.DataFrame, verbose: bool = False):
        """
        Used for fitting each of the SuperLearners
        :param data:  a pandas dataframe
        :param verbose: boolean, used to see extra information whilst running
        :return:
        """

        for var in self.models.keys():
            var_type = self.var_types[var]
            if verbose:
                print("Training model for var:", var)
            pred_vars = self.predictors[var]
            y = data[var]
            X = data[pred_vars]
            self.models[var].fit(
                X=X, y=y
            )  # the SL library includes the k-fold splitting process

            val_preds = self.models[var].predict(X)
            if var_type == "bin":
                val_preds = (val_preds >= 0.5).astype("int")
            elif var_type == "cat":
                one_hot = np.zeros_like(val_preds)
                one_hot[np.arange(len(val_preds)), np.argmax(val_preds, axis=1)] = 1
                val_preds = one_hot

            if verbose:
                mets = (
                    self.regression_metrics.keys()
                    if var_type == "cont"
                    else self.classification_metrics.keys()
                )
                for metric in mets:
                    result = (
                        self.regression_metrics[metric](y, val_preds)
                        if var_type == "cont"
                        else self.classification_metrics[metric](y, val_preds)
                    )
                    print(metric, ":", result)

            if self.save_models:
                # TODO: mofidy for superlearner
                assert NotImplementedError

    def predict(self, data: pd.DataFrame, var: Optional[str] = None):
        """
        :param data: pandas dataframe
        :param var: a specific variale to be predicted
        :return: if var is None, return a dictionary of predictions for each outcome variable, else return an array for the specific variable
        """
        if var is None:
            predictions = {}

            for model in self.models.keys():
                rdf = self.models[model]
                predictors = self.predictors[model]

                predictions[model] = rdf.predict(data[predictors])

        elif var is not None:
            predictions = self.models[var].predict(data[self.predictors[var]])

        return predictions

    def infer(
        self,
        data: pd.DataFrame,
        intervention_nodes_vals: Optional[Dict[str, float]] = None,
    ):
        """
        This function iterates through the causally-constrained set of SuperLearners according
         to the desired intervention
        :param data is pandas dataframe to accompany the desired interventions (necessary for the variables
         which are not downstream of the intervention nodes). Assumed ordering is topological.
        :param intervention_nodes_vals: dictionary of variable names as strings for intervention with
        corresponding intervention values
        :return: an updated dataset with new values including interventions and effects
        """

        d_temp = data.copy()
        if intervention_nodes_vals is not None:
            # modify the dataset with the desired intervention values
            for var_name in intervention_nodes_vals.keys():
                val = intervention_nodes_vals[var_name]
                d_temp[var_name] = val

            # find all descendants of intervention variables which are not in intervention set
            all_descs = []
            for var in intervention_nodes_vals.keys():
                all_descs.append(list(nx.descendants(self.dag, var)))
            all_descs = [item for sublist in all_descs for item in sublist]
            vars_to_update = set(all_descs) - set(intervention_nodes_vals.keys())  # type: ignore
            # sort variables according to causal ordering
            sorted_vars_to_update = sorted(vars_to_update, key=self.causal_ordering.get)

            # iterate through the dataset / predictions, updating the input dataset each time, where appropriate
            min_int_order = min(
                [self.causal_ordering[var] for var in intervention_nodes_vals.keys()]
            )
            for i, var in enumerate(sorted_vars_to_update):  # type: ignore
                if (
                    self.causal_ordering[var] >= min_int_order
                ):  # start at the causal ordering at least as high as the lowest order of the intervention variable
                    # generate predictions , updating the input dataset each time
                    preds = self.predict(
                        data=d_temp, var=var
                    )  # get prediction for each variable
                    d_temp[
                        var
                    ] = preds  # overwrite variable values with predictions - this gets recycled and used for subsequent predictions
        else:
            d_temp = self.predict(data=data)

        return d_temp

    def get_0_1_ATE(self, data: pd.DataFrame):
        """
        For each possible outcome, this function estimates the ATE of each parent on this outcome when setting the value to 0 and 1
        :param data: a pandas dataframe
        :return: a dictionary of ATEs for each parent-outcome pair
        """

        ATEs = {}
        for outcome in self.models.keys():  # for each outcome
            parents = self.predictors[outcome]
            for parent in parents:  # for each parent of each outcome
                d_temp = data.copy()  # reset the dataframe to the original
                d_temp[parent] = 0  # intervene on parent  (0)
                preds0 = self.predict(
                    data=d_temp, var=outcome
                )  # generate predictions under intervention
                d_temp[parent] = 1  # intervene on parent  (1)
                preds1 = self.predict(
                    data=d_temp, var=outcome
                )  # generate predictions under intervention

                diff = preds1 - preds0  # compute CATE
                est_ATE = diff.mean()  # compute ATE

                ATEs[str(parent + "->" + outcome)] = est_ATE  # add to dictionary
        return ATEs

    def load_models(self, model_dir: Union[str, Path]):
        for var in self.models.keys():
            self.models[var] = joblib.load(
                os.path.join(model_dir, "{}.joblib".format(var))
            )


def bootstrapper(
    num_bootstraps: int,
    subsample_size: int,
    data: pd.DataFrame,
    dag: nx.DiGraph,
    var_types: Dict[str, str],
    int_nodes_val: Optional[Dict[str, float]] = None,
    int_nodes_valb: Optional[Dict[str, float]] = None,
    seed: int = 42,
    learner_list: Optional[List[str]] = None,
    baseline: bool = False,
    k: int = 5,
):
    """A bootstrapping function which returns a set of estimated interventional datasets.
    :param num_bootstraps: (int) the number of bootstrap trials
    :param subsample_size: (int) the number of datapoints to use in a bootstrap subsample
    :param k: (int) number of k-fold cross validation folds
    :param data: (pandas dataframe) data to perform bootstrapping with
    :param dag: (networkx digraph object) the DAG to use
    :param var_types (python dictionary) the variable types ('cont','cat', 'bin')
    :param int_nodes_val: (python dictionary) of variable names and desired intervention [optional]
    :param learner_list: a list of learners for the superlearner
    :param int_nodes_val_b: (python dictionary) of variable names and desired intervention for a contrast [optional]
    """
    np.random.seed(seed)

    if int_nodes_valb is not None:
        assert (
            int_nodes_val is not None
        ), "If a contrast is given, the initial intervention is also needed."

    int_dfs = {}

    for i in tqdm.tqdm(range(num_bootstraps)):
        bootstrap_data = data.sample(n=subsample_size, replace=True)

        daglearner = DAGLearner(
            dag=dag,
            k=k,
            var_types=var_types,
            learner_list=learner_list,
            baseline=baseline,
        )
        daglearner.fit(data=bootstrap_data, verbose=False)

        if int_nodes_val is None:  # if no specific interventions are specified
            ATEs = daglearner.get_0_1_ATE(data=bootstrap_data)
            int_dfs[i] = ATEs
        elif int_nodes_val is not None:
            interventional_dataset = daglearner.infer(
                data=bootstrap_data, intervention_nodes_vals=int_nodes_val
            )
            int_dfs[i] = interventional_dataset

            if int_nodes_valb is not None:
                interventional_dataset1b = daglearner.infer(
                    data=bootstrap_data, intervention_nodes_vals=int_nodes_valb
                )
                int_dfs[i] = interventional_dataset1b - interventional_dataset

    return int_dfs
