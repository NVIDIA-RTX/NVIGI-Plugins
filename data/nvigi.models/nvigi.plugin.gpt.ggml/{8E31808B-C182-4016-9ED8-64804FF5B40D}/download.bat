pushd "%~dp0"
curl -L "https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/ucs-ms/nemotron-mini-4b-instruct/0.1.3.1/files?redirect=true&path=nemotron-mini-4b-instruct-AIM-GGUF.zip" -o out.zip
tar -x -f out.zip -O "{8E31808B-C182-4016-9ED8-64804FF5B40D}/nemotron-4-mini-4b-instruct_q4_0.gguf" > nemotron-4-mini-4b-instruct_q4_0.gguf
tar -x -f out.zip -O "{8E31808B-C182-4016-9ED8-64804FF5B40D}/NVIDIA Software and Model Evaluation License Agreement (2024.06.28).txt" > "NVIDIA Software and Model Evaluation License Agreement (2024.06.28).txt"
del out.zip
popd
IF NOT "%1"=="-nopause" (
	pause
)
