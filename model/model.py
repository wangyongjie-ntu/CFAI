"""Module containing an interface to trained PyTorch model."""

from model.base_model import BaseModel
import torch

class Model(BaseModel):

    def __init__(self, model=None, model_path=''):
        """Init method

        :param model: trained PyTorch Model.
        :param model_path: path to trained model.
        """

        super().__init__(model, model_path)

    def load_model(self):
        if self.model_path != '':
            self.model = torch.load(self.model_path)

    def get_output(self, input_tensor):
        return self.model(input_tensor).float()

    def set_eval_mode(self):
        self.model.eval()

    def get_gradient(self, input):
        # Future Support
        raise NotImplementedError("Future Support")
