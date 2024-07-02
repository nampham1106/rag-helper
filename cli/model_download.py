from optimum.onnxruntime import ORTModelForFeatureExtraction
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

def export_to_onnx(model_id: str, onnx_path: str):
    model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.save_pretrained(onnx_path)
    tokenizer.save_pretrained(onnx_path)

def dynamic_quantized(model_id: str, onnx_path: str):
    ort_model = ORTModelForFeatureExtraction.from_pretrained(onnx_path)
    quantizer = ORTQuantizer.from_pretrained(ort_model)

    dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)   
    model_quantized_path = quantizer.quantize(
        save_dir=onnx_path,
        quantization_config=dqconfig
    )   
    return model_quantized_path


if __name__ == "__main__":
    model_id = "intfloat/multilingual-e5-large"
    onnx_path = "data/onnx"
    # export_to_onnx(model_id, onnx_path)
    print(dynamic_quantized(model_id, onnx_path))

