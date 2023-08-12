# SLEM - Super Learner Equation Modeling

SLEM / DAGLearner takes in some data, a DAG (as a networkx graph), and a dictionary of variable types, and implements Super Learners for every variable with parents in that DAG.

This enables one to perform effect-size estimation without the usual constraints on functional form associated with (linear) path models / structural equation models.

What users need:

- a dataset (pandas dataframe)
- a dag (networkx DiGraph object)
- a dictionary of variable types (continuous='cont', binary='bin', categorical='cat')

What SLEM/DAGLearner does:

- estimates all causal links with SuperLearners (or optional linear/logist baseline)
- provides ATE estimates for all paths in the model
- provides estimates for any given intervention


Options:

- Users can specify a list of candidate learners according to the following options:
  - 'LR': Linear or logistic regression (depending on variable type)
  - 'RF': Random Forest
  - 'Elastic': Elastic Net
  - 'MLP': MultiLayer Perceptron
  - 'SV': Support Vector Machine
  - 'AB': AdaBoost
  - 'BR': Bayesian Ridge or Naive Bayes
  - 'poly': polynomial regression
  - e.g. ```daglearner = DAGLearner(dag=DAG,  var_types=var_types, k=6, learner_list=['AB','LR'])```
- If a list of such learners is not specified, all will be used
- The number of folds in the k-fold superlearner fitting process k


DAGLearner includes the following methods:

- ```daglearner.fit(data=df, verbose=True)``` (for fitting the superlearner to a given dataset)
- ```daglearner.get_0_1_ATE(data=df)``` for extracting all average path coefficients for do(0) vs. do(1) interventions 
- ```daglearner.predict(data=df, var='Y')``` for generating predictions for a specific variable
- ```daglearner.infer(data=df, intervention_nodes_vals={'X': 0, 'C':0})``` for generating (causal) predictions for a specific intervention


The ```example_usage.ipynb``` includes ways to integrate it into a pipeline with bootstrapping and plotting.


# Quickstart:

In terminal (e.g. with specific venv or conda environment):
```bash 
conda create -n slem_env -y python=3.9 
conda activate slem_env
pip3 install slem-learn
```


In Python:

```python
from src.slem import DAGLearner
import pandas as pd
import networkx as nx
import numpy as np

# generate data
ux = np.random.randn(500, 1)
uc = np.random.randn(500, 1)
uy = np.random.randn(500, 1)

C = uc
X = 0.7 * C + ux
Y = 0.7 * X + 0.6 * C + uy

# put data in pandas dataframe
df = pd.DataFrame(np.concatenate([C, X, Y], 1))
df.columns = ['C', 'X', 'Y']

# specify variable types
var_types = {'C': 'cont', 'X': 'cont', 'Y': 'cont'}
# specify DAG
DAG = nx.DiGraph()
DAG.add_edges_from([('C', 'Y'), ('C', 'X'), ('X', 'Y')])

daglearner = DAGLearner(dag=DAG, var_types=var_types, k=6)
# fit the superlearners:
daglearner.fit(data=df, verbose=True)
# estimate all ATEs for do(0) and do(1) interventions:
ATEs = daglearner.get_0_1_ATE(data=df)

```