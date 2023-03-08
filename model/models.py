
def create_model(opt):
    print(opt.model)
    if opt.model == 'DeFLOCNet':
        from .DeFLOCNet import DeFLOCNet
        model = DeFLOCNet(opt)
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    print("model [%s] was created" % (model.name()))
    return model
