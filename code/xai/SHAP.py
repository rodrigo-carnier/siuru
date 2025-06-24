import shap
import logging
import matplotlib.pyplot as plt
from xai.IExplainableAIAlgorithm import IExplainableAIAlgorithm

log = logging.getLogger()

class SHAP(IExplainableAIAlgorithm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def explainer(self,
                model,
                train_data,
                feature_names,
                class_names,
                **kwargs
        ):
        # Extract the sklearn model
        if hasattr(model, 'model_instance'):
            model = model.model_instance

        # Ensure model is SHAP compliant.
        if not callable(getattr(model, "predict", None)):
            raise TypeError(f"The passed model is not callable and cannot be analyzed directly with the given masker! Model: {model}")
      
        shap_values = shap.TreeExplainer(model).shap_values(train_data)
        shap.summary_plot(shap_values, train_data, feature_names= feature_names, class_names=class_names, max_display=30, plot_size=[10,7])
        shap_fig = plt.gcf()
        
        return shap_fig


