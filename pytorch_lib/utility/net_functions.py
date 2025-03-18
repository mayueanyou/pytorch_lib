def count_parameters(model, trainable=False):
    if trainable: return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def freeze_model(net,layers=None):
    if layers is None:
        for param in net.parameters(): param.requires_grad = False
    else:
        for layer in layers:
            for param in getattr(net, layer).parameters(): param.requires_grad = False

def unfreeze_model(net,layers=None):
    if layers is None:
        for param in net.parameters(): param.requires_grad = True
    else:
        for layer in layers:
            for param in getattr(net, layer).parameters(): param.requires_grad = True

def check_layer_status(net):
    layer_info = {}
    for layer_name, module in net.named_modules():
        layer_info[layer_name] = {}
        for name,param in module.named_parameters():
            layer_info[layer_name][name] = param.requires_grad
    return layer_info