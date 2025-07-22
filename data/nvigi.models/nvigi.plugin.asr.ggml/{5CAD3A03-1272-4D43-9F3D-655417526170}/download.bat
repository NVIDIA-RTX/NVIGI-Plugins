pushd "%~dp0"
curl -L -o whisper-small.gguf "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin?download=true"
popd
IF NOT "%1"=="-nopause" (
	pause
)