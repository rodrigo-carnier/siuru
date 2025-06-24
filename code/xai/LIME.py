import lime
from lime import lime_tabular
from xai.IExplainableAIAlgorithm import IExplainableAIAlgorithm


class LIME(IExplainableAIAlgorithm):
    def __init__(self, **kwargs):
        self.explainer_value = None
        super().__init__(**kwargs)

    def explainer(self,
                train_data,
                feature_names,
                class_names,
                **kwargs
        ):

        self.explainer_value = lime.lime_tabular.LimeTabularExplainer(train_data, feature_names=feature_names, class_names=class_names, verbose=False)

    def explainIntance(self,
                test_data,
                predict_proba,
                feature_names,
                **kwargs
        ):

        exp = self.explainer_value.explain_instance(test_data, predict_proba, num_features=len(feature_names))

        return exp
        
