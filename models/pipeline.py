"""
Enhanced Pipeline for Financial ML (AFML Snippet 9.2)
Overrides sklearn.pipeline.Pipeline.fit() to correctly intercept sample_weight
and inject it into the final estimator's fit parameters.
"""
from sklearn.pipeline import Pipeline


class MyPipeline(Pipeline):
    """
    Custom Pipeline implementation that intercepts sample_weight and routes
    it directly to fit_params mapped to the final estimator step.
    """
    def fit(self, X, y=None, sample_weight=None, **fit_params):
        if sample_weight is not None:
            step_name = self.steps[-1][0]
            fit_params[f"{step_name}__sample_weight"] = sample_weight
        return super(MyPipeline, self).fit(X, y, **fit_params)
