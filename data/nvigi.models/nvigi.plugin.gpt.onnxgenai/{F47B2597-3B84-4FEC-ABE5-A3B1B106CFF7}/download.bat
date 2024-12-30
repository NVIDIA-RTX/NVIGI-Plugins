pushd "%~dp0"
curl -L -o genai_config.json "https://api.ngc.nvidia.com/v2/models/nvidia/nvigisdk/mistral_7b/versions/1.0/files/genai_config.json"
curl -L -o License.txt "https://api.ngc.nvidia.com/v2/models/nvidia/nvigisdk/mistral_7b/versions/1.0/files/License.txt"
curl -L -o model.onnx "https://api.ngc.nvidia.com/v2/models/nvidia/nvigisdk/mistral_7b/versions/1.0/files/model.onnx"
curl -L -o model.onnx.data "https://api.ngc.nvidia.com/v2/models/nvidia/nvigisdk/mistral_7b/versions/1.0/files/model.onnx.data"
curl -L -o special_tokens_map.json "https://api.ngc.nvidia.com/v2/models/nvidia/nvigisdk/mistral_7b/versions/1.0/files/special_tokens_map.json"
curl -L -o tokenizer.json "https://api.ngc.nvidia.com/v2/models/nvidia/nvigisdk/mistral_7b/versions/1.0/files/tokenizer.json"
curl -L -o tokenizer_config.json "https://api.ngc.nvidia.com/v2/models/nvidia/nvigisdk/mistral_7b/versions/1.0/files/tokenizer_config.json"
popd
pause
