
def build_network(cfg, test=False):
    if cfg.model_name == 'semseg':
        from .semseg.semseg import SemSeg as Network, model_fn_decorator
    else:
        raise NotImplementedError(f'{cfg.model_name} not implemented.')

    model = Network(cfg)
    model_fn = model_fn_decorator(cfg, test)

    return model, model_fn