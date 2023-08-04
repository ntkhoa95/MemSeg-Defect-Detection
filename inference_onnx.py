# Importing libraries
import torch, cv2
import numpy as np
import os, yaml, time
import onnx, onnxruntime
from timm import create_model
from data import create_dataset
from models import MemSeg
from scipy.signal import medfilt2d
import torchvision.transforms as transforms

import warnings
warnings.filterwarnings("ignore")

# Dataset
def load_dataset(object_name):
    global testset
    testset = create_dataset(
                        datadir                = cfg['DATASET']['datadir'],
                        target                 = object_name, 
                        train                  = False,
                        resize                 = cfg['DATASET']['resize'],
                        texture_source_dir     = cfg['DATASET']['texture_source_dir'],
                        structure_grid_size    = cfg['DATASET']['structure_grid_size'],
                        transparency_range     = cfg['DATASET']['transparency_range'],
                        perlin_scale           = cfg['DATASET']['perlin_scale'], 
                        min_perlin_scale       = cfg['DATASET']['min_perlin_scale'], 
                        perlin_noise_threshold = cfg['DATASET']['perlin_noise_threshold']
                        )

# Model
def load_model(object_name):
    global model, cfg
    cfg = yaml.load(open(f'./configs/{object_name.split("-")[-1]}.yaml','r'), Loader=yaml.FullLoader)
    onnx_path = "/mnt/ssd500/vicell/anomaly_inspection/MS/export_onnx/engine.onnx"
    model = onnxruntime.InferenceSession(onnx_path)

    return model

def minmax_scaling(img):
    return (((img - img.min()) / (img.max() - img.min())) * 255).to(torch.uint8)

def heatmap_on_image(heatmap, image):
    if heatmap.shape != image.shape:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    out = np.float32(heatmap)/255 + np.float32(image)/255
    out = out / np.max(out)
    return np.uint8(255 * out)

def processing_input(img, target_size=(256, 256)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=target_size)
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225])
        ])
    img = transform(img)
    img = np.asarray(img.unsqueeze(0))
    print(img.shape)

    return img

def predict(idx):
    input_i, mask_i, target_i = testset[idx] # input_i: Tensor([3, 256, 256])
    input_i = np.asarray(input_i.unsqueeze(0))
    ort_input = {model.get_inputs()[0].name: input_i}
    output_i = model.run(None, ort_input)[0]
    output_i = torch.from_numpy(output_i)

    output_i = torch.nn.functional.softmax(output_i, dim=1)
    input_i = minmax_scaling(input_i.permute(1, 2, 0))

    return input_i, output_i[0][1]

def inference(img, min_area=15):
    ort_input = {model.get_inputs()[0].name: img}
    output = model.run(None, ort_input)[0]
    output = torch.from_numpy(output)
    # output = output.unsqueeze(0).cuda().cpu().detach()
    output = torch.nn.functional.softmax(output, dim=1)[0][1]
    output = cv2.normalize(np.array(output), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, output_binary = cv2.threshold(output, 0, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(output_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours_image = np.zeros((output_binary.shape[0], output_binary.shape[1]))
    
    filter_contours = [contours[i] for i in range(len(contours)) if min_area <= cv2.contourArea(contours[i])]
    cv2.drawContours(contours_image, filter_contours, -1, 255, -1)
    contours_image = medfilt2d(contours_image, 5)
    output_heatmap = cv2.applyColorMap(np.uint8(contours_image), cv2.COLORMAP_JET)
    output_heatmap_bgr = cv2.cvtColor(output_heatmap, cv2.COLOR_RGB2BGR)

    return filter_contours, output_heatmap_bgr
    # return output, output

def inference_ui(img):
    img = cv2.resize(img, (256, 256))
    # input = processing_input(img)
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225])
        ])
    input_tensor = transform(img)
    filter_contours, output_heatmap_bgr = inference(input_tensor)
    hm_on_img = heatmap_on_image(output_heatmap_bgr, img)
    cv2.drawContours(hm_on_img, filter_contours, -1, (255, 0, 0), 5)
    # hm_on_img_bgr = cv2.cvtColor(hm_on_img, cv2.COLOR_BGR2RGB)
    # hm_on_img = cv2.resize(hm_on_img, img.shape[:2])
    return hm_on_img

if __name__ == "__main__":

    # model_list = 'MemSeg-Original-pad'
    model_list = 'MS-pcb'
    print("Loading model ...")
    model = load_model(object_name=model_list)

    print("Loaded model ...")
    obj = model_list.split("-")[-1]

    output_path = f"./samples/{model_list}"
    os.makedirs(output_path, exist_ok=True)

    mode = "test_folder" # TODO: change mode for testing ["test_single", "test_dataset", "test_folder"]

    print("Starting infer ...")
    if mode == "test_single": # inference single input image
        fold = "crack"
        img_path = f"./datasets/MVTec/{obj}/test/{fold}/017.png"
        img_name = img_path.split("/")[-1].split(".")[0]
        tik = time.time()
        img = cv2.imread(img_path)
        input_tensor = processing_input(img)
        filter_contours, output_heatmap_bgr = inference(input_tensor)
        
        hm_on_img = heatmap_on_image(output_heatmap_bgr, img)
        hm_on_img_bgr = cv2.cvtColor(hm_on_img, cv2.COLOR_BGR2RGB)

        contours_image = np.zeros((img.shape[0], img.shape[1]))
        cv2.drawContours(contours_image, filter_contours, -1, 255, -1)

        cv2.imwrite(os.path.join(output_path, f"{img_name}_{fold}_in.png"), img)
        cv2.imwrite(os.path.join(output_path, f"{img_name}_{fold}_out.png"), contours_image)
        cv2.imwrite(os.path.join(output_path, f"{img_name}_{fold}_heatmap.png"), hm_on_img_bgr)
        tok = time.time()
        print("Processing time:\t", round(tok-tik, 5))

    elif mode == "test_folder":
        print("Processing folder:\t", obj)
        phase = "inference"
        fold = "broken"
        local_path = f"./datasets/MVTec/{obj}/{phase}/{fold}"
        output_path = os.path.join(output_path, obj, phase, fold)
        os.makedirs(output_path, exist_ok=True)
        img_list = sorted(os.listdir(local_path))
        for img_name in img_list:
            name = img_name.split(".")[0]
            tik = time.time()
            img = cv2.imread(os.path.join(local_path, img_name))
            input_tensor = processing_input(img)
            filter_contours, output_heatmap_bgr = inference(input_tensor)
            
            cv2.drawContours(output_heatmap_bgr, filter_contours, -1, (0, 255, 0), 1)
            hm_on_img = heatmap_on_image(output_heatmap_bgr, img)
            hm_on_img_bgr = cv2.cvtColor(hm_on_img, cv2.COLOR_BGR2RGB)
            
            contours_image = np.zeros((output_heatmap_bgr.shape[0], output_heatmap_bgr.shape[1]))
            cv2.drawContours(contours_image, filter_contours, -1, 255, -1)
            contours_image = cv2.resize(contours_image, (img.shape[1], img.shape[0]))
            contours_image[contours_image != 0] = 255
        
            cv2.imwrite(os.path.join(output_path, f"{name}_in.png"), img)
            cv2.imwrite(os.path.join(output_path, f"{name}_out.png"), contours_image)
            cv2.imwrite(os.path.join(output_path, f"{name}_heatmap.png"), hm_on_img_bgr)
            tok = time.time()
            print("Processing time:\t", round(tok-tik, 5), img_name)        

    elif mode == "test_dataset": # inference iter loaded dataset
        load_dataset(object_name=model_list.split('-')[-1])
        for idx in range(10):
            tik = time.time()
            input_tensor, output = predict(idx)
            
            output = cv2.normalize(np.array(output), None, 0, 255, cv2.NORM_MINMAX)
            output_heatmap = cv2.applyColorMap(np.uint8(output), cv2.COLORMAP_JET)
            output_heatmap_bgr = cv2.cvtColor(output_heatmap, cv2.COLOR_RGB2BGR)

            input_tensor = cv2.normalize(np.array(input_tensor), None, 0, 255, cv2.NORM_MINMAX)
            input_bgr = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)

            hm_on_img = heatmap_on_image(output_heatmap_bgr, input_bgr)
            hm_on_img_bgr = cv2.cvtColor(hm_on_img, cv2.COLOR_BGR2RGB)

            cv2.imwrite(os.path.join(output_path, f"{idx}_out.png"), output)
            cv2.imwrite(os.path.join(output_path, f"{idx}_heatmap.png"), hm_on_img_bgr)
            cv2.imwrite(os.path.join(output_path, f"{idx}_in.png"), input_bgr)
            tok = time.time()
            print("Processing time:\t", round(tok-tik, 5))

    elif mode == "test_ui":
        import gradio as gr
        gr.Interface(fn=inference_ui, inputs="image", outputs="image").launch()
