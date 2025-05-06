pushd "%~dp0"
curl -L -o AsqFlowGeneratorModel_onnx19_float16_trt16.engine "https://api.ngc.nvidia.com/v2/models/org/nvidia/team/nvigisdk/riva-tts-a2flow/1.3/files?redirect=true&path=AsqFlowGeneratorModel_onnx19_float16_trt16.engine"
curl -L -o bigvgan_VocoderModel_onnx16_float16_trt16.engine "https://api.ngc.nvidia.com/v2/models/org/nvidia/team/nvigisdk/riva-tts-a2flow/1.3/files?redirect=true&path=bigvgan_VocoderModel_onnx16_float16_trt16.engine"
curl -L -o DpModel_float16_trt16.engine "https://api.ngc.nvidia.com/v2/models/org/nvidia/team/nvigisdk/riva-tts-a2flow/1.3/files?redirect=true&path=DpModel_float16_trt16.engine"
curl -L -o G2P_onnx16.onnx "https://api.ngc.nvidia.com/v2/models/org/nvidia/team/nvigisdk/riva-tts-a2flow/1.3/files?redirect=true&path=G2P_onnx16.onnx"
curl -L -o ipa_dict_phonemized.txt "https://api.ngc.nvidia.com/v2/models/org/nvidia/team/nvigisdk/riva-tts-a2flow/1.3/files?redirect=true&path=ipa_dict_phonemized.txt"
curl -L -o tokenize_and_classify.far "https://api.ngc.nvidia.com/v2/models/org/nvidia/team/nvigisdk/riva-tts-a2flow/1.3/files?redirect=true&path=tokenize_and_classify.far"
curl -L -o verbalize.far "https://api.ngc.nvidia.com/v2/models/org/nvidia/team/nvigisdk/riva-tts-a2flow/1.3/files?redirect=true&path=verbalize.far"
popd
pause
