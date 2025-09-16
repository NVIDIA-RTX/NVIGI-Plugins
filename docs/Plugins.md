# NVIDIA In-Game Inference AI Plugins

In-Game Inferencing (NVIGI) is an open-sourced cross platform solution that simplifies integration of the latest NVIDIA and other provider's technologies into applications and games. This framework allows developers to easily implement one single integration and enable multiple technologies supported by the hardware vendor or cloud provider. Supported technologies include AI inference but, due to the generic nature of the NVIGI SDK, can be expanded to graphics or any other field of interest.

For high-level details of NVIGI, as well as links to the official downloads for the product, see [NVIDIA In-Game Inferencing](https://developer.nvidia.com/rtx/in-game-inferencing)

This directory contains the main AI plugins for integrating In-Game Inferencing-based AI workflows into your application.

> **IMPORTANT:**
> For important changes and bug fixes included in the current release, please see the release notes at the end of this document BEFORE use.

## Prerequisites

### Hardware

- NVIDIA RTX 30x0/A6000 series or (preferably) RTX 4080/4090 or RTX 5080/5090 with a minimum of 8GB and recommendation of 12GB VRAM.  

> NOTE: Some plugins only support RTX 40x0 and above, and will not be available on RTX 30x0.

### Windows

- Win10 20H1 (version 2004 - 10.0.19041) or newer
- Install graphics driver r555.85 or newer
- Install VS Code or VS2019/VS2022 with [SDK 10.0.19041+](https://go.microsoft.com/fwlink/?linkid=2120843)

### NVIGI Components

NVIGI consists of several independent components, of which this SDK directory is only one.  Normally, several of these components are used together, and in many cases, developers will actually receive a combined, "standard-layout" pack that is easier to use out of the box.  This SDK directory contains plugins focused upon AI inference, especially speech recognition, large language models, and embeddings.

The SDK Plugins pack/tree rely upon and require the NVIGI core APIs, which are shipped as the `nvigi_core` Plugin Development Kit (PDK) or Runtime Pack.  The two packs have different use-cases with the plugins:
- Applications that want to use NVIGI-based AI features will need the NVIGI Core Runtime Pack, which contains the core DLLs and other runtime components needed to build and run NVIGI-based applications along with pre-built plugins.
- Developers who want to build their own plugins need the NVIGI Core PDK.

These packs can be found either as:
- **Runtime Pack:** A part of any NVIGI binary SDK pack, downloadable fom [NVIDIA In-Game Inferencing](https://developer.nvidia.com/rtx/in-game-inferencing).  In this case, the NVIGI Core Runtime Pack will be located in a subdirectory of this SDK directory called `nvigi_core`
- **PDK:** A pre-built download of the NVIGI Core PDK, downloadable from the [NVIGI Source Core Repo](https://github.com/NVIDIA-RTX/NVIGI-core) as a release artifact with each release.
- **PDK or Runtime Pack:** A locally cloned and built copy of the `nvigi_core` git repository.  See the [NVIGI Source Root Repo](https://github.com/NVIDIA-RTX/NVIGI) for instructions on how to build and package the core. 

NVIGI core is documented in detail in its own documentation, which should be available in any distribution.  Developers should read the documentation in `nvigi_core` to gain a basic understanding of the overall architecture into which the plugins in this source tree plug into.

## DIRECTORY SETUP

As noted, building the NVIGI plugins SDK from source requires access to a fully-built local copy of the NVIGI Core PDK.  We will call the location of this PDK `<NVIGI_CORE>`.

In the following instructions,
- `<SDK_PLUGINS>` should be replaced with the full path to your NVIGI SDK directory (the path of this README).  In binary packs, this is the root directory.
- `<NVIGI_CORE>` should be replaced with the fully path to the NVIGI Core PDK, either pulled pre-built or built from git source.  See above for details.  In binary packs, this is `<SDK_PLUGINS>/nvigi_core`.
- `<SDK_MODELS>` refers to the location of the `nvigi.models` AI models directory.
  - This is available in `<SDK_PLUGINS>/data/nvigi.models`
- `<SDK_TEST>` refers to the location of the `nvigi.test` AI test data directory.
  - This is available in `<SDK_PLUGINS>/data/nvigi.test`

### Required connection from the SDK to the Core

The NVIGI Core is a distinct NVIGI component, and available for download separately from this NVIGI plugins. The core build required depends upon whether the developer intends to use pre-built plugins with their app (Core Runtime) or intends to build their own plugins and the NVIDIA plugins from source (Core PDK).  But, in some cases, core may be shipped or pulled with another distribution to make it easier for developers to get started exploring NVIGI.  For example:

- In the binary packs, the NVIGI Core Runtime Pack is included in a subdirectory of the SDK directory called `nvigi_core`.
- In source releases, the core source is its own repo, which is cloned separately from the plugins/SDK repo and a jun link is created from `<SDK_PLUGINS>/nvigi_core` to the cloned core repo.

It can be quickly determined if this is the case by looking at the contents of the `<SDK_PLUGINS>` directory, and checking if a directory called `<SDK_PLUGINS>/nvigi_core` is present. If so, then this SDK pack already ships with the NVIGI Core, and it is not necessary to download it separately.

If this directory is not present, then developers need to obtain an instance of the NVIGI Core separately (either download a pre-built one or build one from git source). Further, a directory junction link from `<SDK_PLUGINS>/nvigi_core` to `<NVIGI_CORE>` needs to be created; this is because the NVIGI plugin SDK build scripts assume the NVIGI Core will be located at `<SDK_PLUGINS>/nvigi_core`.

This directory junction link can be done easily by running the following command within a VS2022 Development Command Prompt from the `<SDK_PLUGINS>` directory:

`mklink /j nvigi_core <NVIGI_CORE>` 

## Basic Steps: Summary

The following sections detail each of the steps of how to set up, rebuild and run the samples in this pack from the debugger or command line.  But the basic steps are:
1. Generate the project files
   1. Open a VS2022 Development Command Prompt to the SDK (a.k.a. `<SDK_PLUGINS>`) directory
   2. Run `setup.bat`
> NOTE:
> If `setup.bat` fails with an error from the `packman` package downloader, please re-run `setup.bat` again as there are rare but possible issues with link-creation on initial run.

2. Build the project, which will build the plugins and samples from source
   1. Launch VS2022
   2. Load `<SDK_PLUGINS>/_project/vs2022/nvigi.sln` as the workspace
   3. Build the Release configuration
3. Copy the updated binaries into the expected, packaged locations
   1. Open a (or use an existing) VS2022 Development Command Prompt to the SDK (a.k.a. `<SDK_PLUGINS>`) directory
   2. Run `copy_sdk_binaries.bat Release`
4. Run the sample (assumes that the models have been downloaded as per initial setup instructions)
   1. Open a (or use an existing) VS2022 Development Command Prompt to the SDK (a.k.a. `<SDK_PLUGINS>`) directory
   2. Run `bin\x64\nvigi.basic.exe --models data\nvigi.models`

## Setup 

> **IMPORTANT:** 
> As detailed in [an earlier section](#required-connection-from-the-sdk-to-the-core-pdk), there must be either:
> * An instance of the NVIGI Core SDK located at `<SDK_PLUGINS>/nvigi_core` (which is the default for some NVIGI SDK packs), or
> * A directory junction link from `<SDK_PLUGINS>/nvigi_core` to your built and packaged copy of the NVIGI Core PDK, `<NVIGI_CORE>`.

To setup the NVIGI repository for building and running the Samples, open a command prompt window or PowerShell to the `<SDK_PLUGINS>` directory and simply run the following command:

```sh
setup.bat
```

> **IMPORTANT:** 
> These steps are listed in the following section are done for you by the `setup.bat` script - **do not run these steps manually/individually**.  This is merely to explain what is happening for you.

When you run the `setup.bat` script, the script will cause two things to be done for you:

1. The NVIDIA tool `packman` will pull all build dependencies to the local machine and cache them in a shared directory.  Links will be created from `external` in the NVIGI tree to this shared cache for external build dependencies.
2. `premake` will generate the project build files in `_project\vs20XX` (for Windows)

> NOTE:
> If `setup.bat` fails with an error from the `packman` package downloader, please re-run `setup.bat` again as there are rare but possible issues with link-creation on initial run.

## Building

To build the project, simply open `_project\vs20XX\nvigi.sln` in Visual Studio, select the desired build configuration and build.

> NOTE: To build the project minimal configuration is needed. Any version of Windows 10 will do. Then
run the setup and build scripts as described here above. That's it. The specific version of Windows, and NVIDIA driver 
are all runtime dependencies, not compile/link time dependencies. This allows NVIGI to build on stock
virtual machines that require zero configuration.

## "Installing" Built Binaries

Once you build your own binaries, either the SDK DLLs or the Sample executable, those will reside in `_artifacts`, not `bin\x64`.  After building any configuration (Release, Debug, etc), you will need to run the `copy_sdk_binaries.bat <cfg>` script.  If you build the Release config, this would be:

1. Open a command prompt to the SDK root
1. run `copy_sdk_binaries.bat Release`

This will copy the core lib, the DLLs and the exes to `bin\x64` and should be done before trying the next section.

## Samples

Setting up and running the samples as shipped pre-built or built using the instructions above is discussed in detail in [Samples](docs/Samples.md)

## Changing an Existing Project

> IMPORTANT: Do not edit the MSVC project files directly!  Always modify the `premake.lua` or files in `premake`.

When changing an existing project's settings or contents (ie: adding a new source file, changing a compiler setting, linking to a new library, etc), it is necessary to run `setup.bat` again for those changes to take effect and MSVC project files and or solution will need to be reloaded in the IDE.

> NOTE: NVIDIA does not recommend making changes to the headers in `include`, as these can affect the API itself and can make developer-built components incompatible with NVIDIA-supplied components.

## Programming Guides 

- [Automatic Speech Recognition](ProgrammingGuideASRWhisper.md)
- [Embedding](ProgrammingGuideEmbed.md)
- [Generative Pre-Trained Transformer](ProgrammingGuideGPT.md)
- [Text To Speech Riva Magpie-TTS-Flow (was ASquaredFlow)](ProgrammingGuideTTSASqFlow.md)

## Sample App and Source

A 3D-rendered sample application using In-Game Inferencing may be found in most SDK packs.  If the pack is in standard layout, this will be in a sibling directory of `sdk` called `sample`.  There is a `README` in that directory with detailed instructions.

## Release Notes:

