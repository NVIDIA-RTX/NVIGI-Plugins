# First Steps

## Abstract

The target audience for this document are developers interested in:
- Exploring the capabilities of this SDK,
- Assess how well its features could address the AI inference needs of their videogame, game engine or interactive experience, and/or
- Estimate the effort to integrate NVIGI into a large codebase.

Therefore, the goal of this document is to suggest a useful path through the NVIGI documentation to address these specific needs.

## Recommended First Steps

In order to explore the capabilities of the SDK, we suggest the following steps:

### 1. Downloading the AI models

Download all of the models supported out-of-the-box by this NVIGI SDK release pack via the `download_all_data.bat` script, which will require approximately **21GB** of storage.

> Note:
> It is not necessary to download all models in order to use the NVIGI developer pack at integration or deployment time. The script mentioned above downloads all of the models at once only for ease of use while exploring this developer pack.

For more details on the models themselves, please see [Getting Models](NVIGIDeveloperPack.md#getting-models).

### 2. Running the 3D Sample

Once the models are downloaded, the best thing to do is to quickly see them in action. The SDK release pack includes the NVIGI 3D Sample, which shows how different kinds of GPU-accelerated AI inference features can be executed concurrently with an interactive, GPU-rendered 3D animated scene.

To launch the sample, run `<SDK_ROOT>/bin/x64/nvigi.3d.exe`, either by double-clicking the executable in Windows Explorer or by running it from a command prompt. The sample will launch a window with a 3D scene and a UI on the left side of the window.

For more details on setup and usage of the 3D sample, read the [3D Sample](Samples3D.md#the-3d-sample) page. If you wish to try the NVIGI AI plugins that perform inference by connecting to a cloud instance, follow the instructions in [Setting up the GPT Cloud Plugin](Samples3D.md#setting-up-the-gpt-cloud-plugin).

### 3. Rebuild the 3D Sample from source, and study its source

Once you have executed the 3D sample, the next step that is recommended is to look at the development setup process for the NVIGI SDK, as well as build the samples included from source. This would allow a developer to focus on the details of integration of the NVIGI SDK, as well as being able to set breakpoints in the code and inspect its execution flow.

For a quick view of the steps to build the samples from source, read [Basic Steps: Summary](Plugins.md#basic-steps-summary). For additional details or troubleshooting, read [Basic Steps: Detailed Breakdown](Plugins.md#basic-steps-detailed-breakdown).

## Next Steps

Once a developer has seen NVIGI inference in action by (1) running the samples and (2) building the plugins from source, some additional suggested steps are:
- Read the documentation on the top-level items: the [models](NVIGIDeveloperPack.md#getting-models), organization of the [docs](NVIGIDeveloperPack.md#important-documentation-in-the-developer-pack), and the [contents](NVIGIDeveloperPack.md#contents-of-the-developer-pack) of the developer pack.
- Read the more general NVIGI [programming guide](ProgrammingGuide.md), which discusses key concepts (i.e. lifecycle, interfaces, input/outputs) and details of an integration of the NVIGI Core API into an existing codebase.
- Read the more specific NVIGI [programming guide for AI](ProgrammingGuideAI.md), which discusses the details of using the NVIGI Core API to load and execute an AI inference plugin.
- Clone the [GitHub repository](https://github.com/NVIDIA-RTX/NVIGI-Plugins) for the NVIGI AI plugins in this developer pack, in order to understand how these existing plugins work.
