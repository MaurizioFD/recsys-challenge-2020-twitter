from Utils.Eval.Metrics import ComputeMetrics


class LGBMImportance:

    def __init__(self, model=None, model_path=None):
        if model_path is not None:
            self.model = self.load_model(model_path)
        elif model is not None:
            self.model = model

    def load_model(self, path):
        from Models.GBM.LightGBM import LightGBM
        model = LightGBM()
        model.load_model(path)
        return model

    def fit(self, *params):
        print("FIT PARAMS")
        print(params)

    def score(self, X_test, Y_test):
        predictions = self.model.get_prediction(X_test)
        cm = ComputeMetrics(predictions, Y_test.to_numpy())
        # Evaluating
        rce = cm.compute_rce()

        print(rce)
        return rce
