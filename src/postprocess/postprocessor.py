from src.module import Module

class Postprocessor(Module):
    def __init__(self, name):
        super().__init__(name)

    def postprocess(self, output):
        raise NotImplementedError
        
    def __call__(self, **kwargs):
        return self.postprocess(**kwargs)