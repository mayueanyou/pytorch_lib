def count_parameters(model):
    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters= sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_parameters = total_parameters - trainable_parameters
    print(f'Total parameters: [{total_parameters:,}], Trainable parameters: [{trainable_parameters:,}], Frozen parameters: [{frozen_parameters:,}]')
    return {'all': total_parameters,'trainable':trainable_parameters,'frozen':frozen_parameters}

def freeze_net(net,layers=None):
    if layers is None:
        for param in net.parameters(): param.requires_grad = False
        print('Freeze whole net!')
    else:
        print(f'Freeze {layers}')
        for layer in layers:
            for param in getattr(net, layer).parameters(): param.requires_grad = False

def unfreeze_net(net,layers=None):
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