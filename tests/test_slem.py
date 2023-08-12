import pytest
import slem
import pandas as pd
import networkx as nx
import numpy as np


def sigm(x):
    return 1 / (1 + np.exp(-x))


def inv_sigm(x):
    return np.log(x / (1 - x))


@pytest.mark.parametrize(
    "baseline",
    [
        (False),
        (True),
    ],
)
def test_slem_ATE(baseline):
    N = 100
    ux = np.random.randn(N, 1)
    uc = np.random.randn(N, 1)
    uy = np.random.randn(N, 1)

    C = uc
    X = 0.7 * C + ux
    Y = 0.7 * X + 0.6 * C + uy

    # put data in pandas dataframe
    df = pd.DataFrame(np.concatenate([C, X, Y], 1))
    df.columns = ["C", "X", "Y"]

    # specify variable types
    var_types = {"C": "cont", "X": "cont", "Y": "cont"}
    # specify DAG
    DAG = nx.DiGraph()
    DAG.add_edges_from([("C", "Y"), ("C", "X"), ("X", "Y")])
    daglearner = slem.DAGLearner(dag=DAG, var_types=var_types, k=4, baseline=baseline)
    daglearner.fit(data=df, verbose=False)
    ATEs = daglearner.get_0_1_ATE(data=df)

    return


@pytest.mark.parametrize(
    "baseline,y_type",
    [
        (False, "cont"),
        (True, "cont"),
        (False, "bin"),
        (True, "bin"),
    ],
)
def test_bootstrapper(baseline, y_type):
    N = 200
    ux = np.random.randn(N, 1)
    uc = np.random.randn(N, 1)
    uy = np.random.randn(N, 1)

    C = uc
    X = 0.7 * C + ux
    if y_type == "cont":
        Y = 0.7 * X + 0.6 * C + uy
    elif y_type == "bin":
        Y = np.random.binomial(1, sigm(0.7 * X + 0.6 * C + uy), (N, 1))

    # put data in pandas dataframe
    df = pd.DataFrame(np.concatenate([C, X, Y], 1))
    df.columns = ["C", "X", "Y"]

    # specify variable types

    var_types = {"C": "cont", "X": "cont", "Y": y_type}
    # specify DAG
    DAG = nx.DiGraph()
    DAG.add_edges_from([("C", "Y"), ("C", "X"), ("X", "Y")])

    bs_results_ATE = slem.bootstrapper(
        num_bootstraps=2,
        subsample_size=50,
        k=4,
        data=df,
        dag=DAG,
        var_types=var_types,
        baseline=baseline,
    )

    int_nodes_val = {"X": 0}
    # by adding one more argument this outputs the interventional dataset for each bootstrap:
    bs_results_int = slem.bootstrapper(
        num_bootstraps=2,
        subsample_size=50,
        k=4,
        int_nodes_val=int_nodes_val,
        data=df,
        dag=DAG,
        var_types=var_types,
        baseline=baseline,
    )

    int_nodes_val = {"X": 0}
    int_nodes_valb = {"X": 1}
    # by adding one more argument this outputs the interventional DIFFERENCE/CONTRAST datasets for each bootstrap:
    bs_results_contrast = slem.bootstrapper(
        num_bootstraps=2,
        subsample_size=50,
        k=4,
        int_nodes_val=int_nodes_val,
        int_nodes_valb=int_nodes_valb,
        data=df,
        dag=DAG,
        var_types=var_types,
        baseline=baseline,
    )
    return
