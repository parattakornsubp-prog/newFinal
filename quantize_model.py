import os
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

def main():
    input_model = "model_final.onnx"
    output_model = "model_quantized.onnx"

    if not os.path.exists(input_model):
        print(f"❌ Error: {input_model} not found in current directory.")
        return

    print(f"🔄 Quantizing {input_model}...")
    
    try:
        # ใช้ Dynamic Quantization (INT8)
        # เหมาะสำหรับโมเดลประเภท Transformer และการรันบน CPU
        quantize_dynamic(
            model_input=input_model,
            model_output=output_model,
            weight_type=QuantType.QUInt8
        )
        
        orig_size = os.path.getsize(input_model) / (1024 * 1024)
        quant_size = os.path.getsize(output_model) / (1024 * 1024)
        
        print(f"✅ Success! Quantized model saved as: {output_model}")
        print(f"📊 Original Size: {orig_size:.2f} MB")
        print(f"📊 Quantized Size: {quant_size:.2f} MB (Reduced by {((orig_size - quant_size) / orig_size) * 100:.1f}%)")
        
    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
