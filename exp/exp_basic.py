import torch
from models import RAT_LLM
class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "RAT_LLM": RAT_LLM,
        }
        self.device = self._acquire_device()
        models = self._build_model()
        if isinstance(models, tuple):
            self.model = tuple(m.to(self.device) for m in models)
        else:
            self.model = models.to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
