def print_grads(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Layer: {name}, Grad: {param.grad}")