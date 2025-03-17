To use the Nemovision 4B Instruct Q4 model, use the following steps.  Note that both the Nemovision 4B Instruct FP16 and Q4 models use the same zipfile.  If you have already set up the other model, you can skip down to step 5 with the zipfile you downloaded previously:

1. Ensure that your NVIDIA NGC account has access to the NVIDIA ACE Early Access (EA) program.  Contact your NVIDIA Developer Relations Manager for more information
2. Log into NGC with the enabled account
3. Navigate to https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ace/models/nemovision-4b-instruct/files (if this shows an error of "This page requires a subscription", ensure that you have logged into the correct NGC account)
4. Download the nemovision-4b-instruct-clip_gguf.zip file - 
5. Open the zip file
6. Unzip the following files from the "gguf" subdirectory into the directory containing this README and nvigi.model.config.json (i.e. without the gguf directory hierarchy)
- minitron-Q4_0.gguf
- mmproj-model-f16.gguf (yes, this is still an FP16 file - that is as intended)
The model can then be loaded via its GUID in an application