import numpy as np
import scipy.stats as stats
import pandas as pd

class DAGFit:
    def __init__(self, daglearner, data, baseline_daglearner=None):
        self.daglearner = daglearner
        self.baseline_daglearner = baseline_daglearner
        self.data = data
        self.observed_cov = None
        self.reproduced_cov = None
        self.df = None

    def compute_reproduced_covariance(self, predictions):
        for key, value in predictions.items():
            if isinstance(value, np.ndarray) and value.ndim > 1:
                if value.shape[1] == 1:
                    predictions[key] = value.ravel()
                else:
                    raise ValueError(f"Cannot handle multi-dimensional predictions for {key}.")
        predictions_df = pd.DataFrame(predictions)        
        reproduced_cov = np.cov(predictions_df, rowvar=False)
        return reproduced_cov

    def chi_squared(self):
        diff = self.observed_cov - self.reproduced_cov
        chi2 = np.trace(np.dot(np.dot(diff, np.linalg.inv(self.reproduced_cov)), diff))
        return chi2

    def srmr(self):
        diff = self.observed_cov - self.reproduced_cov
        return np.sqrt(np.mean(diff**2))

    def rmsea(self, dof):
        chi2 = self.chi_squared()
        rmsea_value = np.sqrt((chi2 - dof) / (dof * self.data.shape[0]))
        return rmsea_value

    def cfi(self, base_chi2):
        chi2 = self.chi_squared()
        return 1 - (chi2 / base_chi2)

    def tli(self, base_chi2, dof, dof_null):
        num = (chi2 / dof) - 1
        denom = (base_chi2 / dof_null) - 1
        return 1 - (num / denom)

    def evaluate(self):
        predictions = self.daglearner.predict(self.data)
        target_variables = list(predictions.keys())
        filtered_data = self.data[target_variables]
        
        self.observed_cov = np.cov(filtered_data, rowvar=False)
        self.reproduced_cov = self.compute_reproduced_covariance(predictions)
        
        p = filtered_data.shape[1]
        k = len(predictions)
        self.df = (p * (p + 1)) / 2 - k

    def compute_metrics(self):
        metrics = {
            'chi_squared': self.chi_squared(),
            'srmr': self.srmr(),
            'rmsea': self.rmsea(dof=self.data.shape[1])
        }
        
        if self.baseline_daglearner:
            baseline_fit = DAGFit(self.baseline_daglearner, self.data)
            baseline_fit.evaluate()
            
            base_chi2 = baseline_fit.chi_squared()
            metrics['cfi'] = self.cfi(base_chi2)
            
            df_null = baseline_fit.data.shape[1]
            metrics['tli'] = self.tli(base_chi2, dof=self.data.shape[1], dof_null=df_null)
            
        return metrics

    def display(self):
        self.evaluate()
        print("Shape of Observed Covariance Matrix:", self.observed_cov.shape)
        print("Shape of Reproduced Covariance Matrix:", self.reproduced_cov.shape)
        print("\nFit Statistics:")
        metrics = self.compute_metrics()
        for key, value in metrics.items():
            print(f"{key}: {value}")


# Use DAGFit to evaluate the model fit
a_fit = DAGFit(daglearner, df)
a_fit.display()

