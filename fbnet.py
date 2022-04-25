
import torch

import numpy as np

from mobile_cv.model_zoo.models.fbnet_v2 import fbnet
from mobile_cv.model_zoo.models.preprocess import get_preprocess

from convert_keras import build_fbnet

device = torch.device('cpu')

import urllib
from PIL import Image 

def _get_input():
    # Download an example image from the pytorch website
    url, filename = (
        "https://github.com/pytorch/hub/blob/master/images/dog.jpg?raw=true",
        "dog.jpg",
    )
    local_filename, headers = urllib.request.urlretrieve(url, filename)
    input_image = Image.open(local_filename)
    return input_image

#torch_model = fbnet("FBNetV2_L2", pretrained=True).to(device)
torch_model = fbnet("fbnet_a", pretrained=True).to(device)

keras_model = build_fbnet(torch_model)


input_image = _get_input()
preprocess = get_preprocess(torch_model.arch_def['input_size'], torch_model.arch_def['input_size'])
input_tensor = preprocess(input_image)
x = input_tensor.unsqueeze(0)

torch_model.eval()

torch_logits = torch_model(x)
keras_logits = keras_model.predict(x.permute([0,2,3,1]).numpy())

torch_logits = torch_logits.detach().numpy()

print("Max error: {:.2}".format(np.max(np.abs(torch_logits-keras_logits))))

