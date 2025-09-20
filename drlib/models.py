import timm, torch.nn as nn

def create_model(name="efficientnet_b3", num_classes=5, pretrained=True):
    model = timm.create_model(name, pretrained=pretrained, in_chans=3, num_classes=num_classes)
    return model
