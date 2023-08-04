import onnxruntime
import json
import pathlib
import shutil
import sys, os
import zipfile
import onnx
import torch
import yaml
import argparse
from models import MemSeg
from timm import create_model


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(
                os.path.join(root, file),
                os.path.relpath(
                    os.path.join(root, file), os.path.join(path, "..")
                ),
            )


def parse_cfg(path):
    if os.path.isfile(path):
        with open(path, "r") as f:
            config = json.load(f)
            return config

    raise FileNotFoundError("Not found config file.")


def main():
    parser = argparse.ArgumentParser(
        description="MemSeg anomaly detection model conversion"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="config file directory",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="checkpoint directory",
    )
    parser.add_argument(
        "--converted_model",
        type=str,
        required=True,
        help="Output directory contain raw model and onnx model",
    )
    parser.add_argument(
        "--fp16",
        type=bool,
        required=False,
        default=False,
        help="onnx model type",
    )
    parser.add_argument(
        "--opset",
        type=int,
        required=False,
        default=13,
        help="onnx opset version",
    )
    parser.add_argument(
        "--dynamic_batch_size",
        type=bool,
        required=False,
        default=False,
        help="onnx opset version",
    )

    device = torch.device('cuda:0')
    args = parser.parse_args()
    # print(args.opset)
    config_path = args.config_path
    checkpoint_path = args.checkpoint_path
    memory_bank_path = os.path.join(checkpoint_path, "memory_bank.pt")
    model_path = os.path.join(checkpoint_path, "best_model.pt")
    converted_model = args.converted_model

    if "json" in config_path:
        config = parse_cfg(config_path)
    elif "yaml" in config_path:
        config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    is_fp16 = args.fp16
    dymamic_batch_size = args.dynamic_batch_size

    memory_bank = torch.load(memory_bank_path)
    
    memory_bank.device = device
    for k in memory_bank.memory_information.keys():
        memory_bank.memory_information[k] = memory_bank.memory_information[k].to(device)

    encoder = create_model(config['MODEL']['feature_extractor_name'],
                           pretrained=True,
                           features_only=True)
    model = MemSeg(memory_module=memory_bank, encoder=encoder)

    print("Loading model weight...")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    # make directory to save onnx model
    pathlib.Path(converted_model).mkdir(parents=True, exist_ok=True)

    if is_fp16:
        x = torch.randn((1, 3, config['DATASET']['resize'][0], config['DATASET']['resize'][1])).cuda().half()
        model = model.half()
    else:
        x = torch.randn((1, 3, config['DATASET']['resize'][0], config['DATASET']['resize'][1])).cuda()

    onnx_model_path = os.path.join(
        converted_model, "engine.onnx"
    )
    model.cuda()
    torch.onnx.export(
        model,
        x,
        onnx_model_path,
        export_params=True,
        opset_version=args.opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        if dymamic_batch_size
        else None,
    )

    print("[INFO]... Finished Conversion ...")
    print("[INFO]... Starting to Check ONNX ...")
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    print("[INFO]... Finished Checking ONNX ...")

    # copy raw model to output model folder
    zip_name = "raw_model.zip"
    zip_path = "raw_model"
    model_zip_path = os.path.join(zip_path, "checkpoint.h5")

    os.makedirs(zip_path, exist_ok=True)
    shutil.copy2(config_path, zip_path)
    shutil.copy2(model_path, model_zip_path)
    print("*** Ziping raw model")
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipdir(zip_path, zipf)

    shutil.copy2(zip_name, converted_model)


if __name__ == "__main__":
    main()
