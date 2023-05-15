import torch
import torch.neuron
import cv2
import numpy as np

from config.load_config import load_yaml, DotDict
from model.craft import CRAFT
from utils.util import copyStateDict
from utils.craft_utils import getDetBoxes
from data import imgproc


if __name__ == "__main__":
    config = load_yaml("main")
    config = DotDict(config)

    model = CRAFT()
    model_path = config.test.trained_model
    net_param = torch.load(model_path)
    model.load_state_dict(copyStateDict(net_param["craft"]))
    model.eval()

    image_path = config.test.custom_data.test_data_dir
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        img, config.test.custom_data.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=config.test.custom_data.mag_ratio
    )
    ratio_h = ratio_w = 1 / target_ratio
    input_tensor = imgproc.normalizeMeanVariance(img_resized)
    input_tensor = torch.from_numpy(input_tensor).permute(2,0,1)
    input_tensor = torch.autograd.Variable(input_tensor.unsqueeze(0))
    print(input_tensor.shape)

    with torch.no_grad():
        y_cpu, feature_cpu = model(input_tensor)
    score_text = y_cpu[0, :, :, 0].cpu().data.numpy().astype(np.float32)
    score_link = y_cpu[0, :, :, 1].cpu().data.numpy().astype(np.float32)
    score_text_cpu = score_text[: size_heatmap[0], : size_heatmap[1]]
    score_link_cpu = score_link[: size_heatmap[0], : size_heatmap[1]]

    boxes_cpu, polys_cpu = getDetBoxes(
        score_text_cpu, score_link_cpu,
        config.test.custom_data.text_threshold,
        config.test.custom_data.link_threshold,
        config.test.custom_data.low_text,
        config.test.custom_data.poly
    )

    convert_neuron = True
    if convert_neuron:
        model_neuron = torch.neuron.trace(model, [input_tensor])
        filename = 'model_neuron.pt'
        model_neuron.save(filename)

    validate_neuron = False
    if validate_neuron:
        model_neuron = torch.jit.load('model_neuron.pt')
        y_neuron, feature_neuron = model_neuron(input_tensor)
        score_text = y_neuron[0, :, :, 0].cpu().data.numpy().astype(np.float32)
        score_link = y_neuron[0, :, :, 1].cpu().data.numpy().astype(np.float32)
        score_text_neuron = score_text[: size_heatmap[0], : size_heatmap[1]]
        score_link_neuron = score_link[: size_heatmap[0], : size_heatmap[1]]
        boxes_neuron, polys_neuron = getDetBoxes(
            score_text_neuron, score_link_neuron,
            config.test.custom_data.text_threshold,
            config.test.custom_data.link_threshold,
            config.test.custom_data.low_text,
            config.test.custom_data.poly
        )
        print(score_text_cpu)
        print(score_text_neuron)
        print(score_link_cpu)
        print(score_link_neuron)
        print(boxes_cpu)
        print(boxes_neuron)