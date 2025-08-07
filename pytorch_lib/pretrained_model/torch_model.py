import os,sys,torch,torchvision

class TorchModel:
    def __init__(self, model_name, weights=None, weights_backbone=None, model_path=None):
        self.model_name = model_name
        self.weights = weights
        self.weights_backbone = weights_backbone
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = getattr(sys.modules['torchvision.models'],"get_model")(
            name = self.model_name,
            weights = self.weights,
            weights_backbone = self.weights_backbone)