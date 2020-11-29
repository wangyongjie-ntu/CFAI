"""Module containing a template class as an interface to ML model.
   Subclasses implement model interfaces for different ML frameworks such as TensorFlow or PyTorch.
   All model interface methods are in dice_ml.model_interfaces"""

class BaseModel:

    def __init__(self, model=None, model_path=''):
        """Init method

        :param model: trained ML Model.
        :param model_path: path to trained model.
        """

        self.model = model
        self.model_path = model_path

    def load_model(self):
        raise NotImplementedError

    def get_output(self):
        raise NotImplementedError

    def get_gradient(self):
        raise NotImplementedError
