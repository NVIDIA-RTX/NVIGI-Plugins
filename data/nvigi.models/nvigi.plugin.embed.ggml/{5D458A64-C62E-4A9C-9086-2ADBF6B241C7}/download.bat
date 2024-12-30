pushd "%~dp0"
curl -L -o e5-large-unsupervised_q4_k_s.gguf "https://api.ngc.nvidia.com/v2/models/nvidia/nvigisdk/e5-large-unsupervised/versions/1.0/files/e5-large-unsupervised_q4_k_s.gguf"
popd
pause
