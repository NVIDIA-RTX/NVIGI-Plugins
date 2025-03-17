# Creating a Customized Plugin

This document guides you through the process of creating a customized NVIGI GPT plugin, based on the original GPT plugin from the SDK. 

**> IMPORTANT: Although the document uses the GPT plugin as an example, the high-level steps also apply to other plugins.**

## Prerequisites

* Familiarity with the basic plugin project setup described in https://github.com/NVIDIA-RTX/NVIGI-Plugins. Experience with successfully building `nvigi.plugin.gpt.ggml.$backend` is a plus.

> NOTE: `$backend` can be any of the available backends like for example `cuda`, `cpu` etc.

## Steps

This section presents one recommended approach to creating a new GPT plugin. Once you understand the project setup process, you can freely make customized changes without strictly following all the steps.

In the following instructions, we follow the directory setup described [here](https://github.com/NVIDIA-RTX/NVIGI-Plugins?tab=readme-ov-file#directory-setup).

- `<SDK_PLUGINS>` should be replaced with the full path to your NVIGI SDK directory (the path of this README).

### 1. Add a New Plugin Project

* Choose a name for your new plugin, such as `mygpt`.

* Prepare the source files for the new plugin by copying from an existing project.

  * Duplicate the folder containing the source code of the original plugin (`<SDK_PLUGINS>/source/plugins/nvigi.gpt`) and rename it to `<SDK_PLUGINS>/source/plugins/nvigi.mygpt`.

  * (Optional) Enter the new source folder and remove unused backend code (e.g., `onnxgenai` and `rest`).

  * (Optional) Rename the source files with your new plugin name by replacing `gpt` with `mygpt`.

  * For example, assuming that we want to modify the GGML backend, the source tree should now have the following structure:

    ```
    <SDK_PLUGINS>/source/plugins
    |-- nvigi.mygpt
    |   |-- ggml
    |   |   |-- premake.lua
    |   |   |-- mygpt.h
    |   |   |-- mygpt.cpp
    |   |   |-- ... // Other source files
    |   |-- nvigi_mygpt.h
    |   |-- // "onnxgenai" and "rest" are removed
    |-- ... // Other plugins
    ```

* Update the setup scripts to include the new source files.

  * Update `NVIGI-Plugins/source/plugins/nvigi.mygpt/ggml/premake.lua` by renaming it with the new plugin name (e.g., rename `group "plugins/gpt"` to `group "plugins/mygpt"` and `project "nvigi.plugin.gpt.ggml.$backend"` to `project "nvigi.plugin.mygpt.ggml.$backend"`).
  * Update `NVIGI-Plugins/premake.lua` to include the project premake file by adding `include("source/plugins/nvigi.mygpt/ggml/premake.lua")` at the end.

* Run `setup.bat` to reflect the changes.

* Open `NVIGI-Plugins/_project/vs2022/nvigi.sln` and check the Solution Explorer - a new project `nvigi.plugin.mygpt.ggml.{$backend}` should be added under `plugins.mygpt`.

### 2. Update Names and GUIDs in Source Files

* Find the NVIGI utility `nvigi.tool.utils.exe` located in `NVIGI-Core\bin\Release_x64`
* Open terminal and run `nvigi.tool.utils.exe --plugin nvigi.plugin.mygpt.ggml.$backend` (make sure to replace `mygpt` and `$backend` accordingly)
* Open `nvigi_mygpt.h`. This is a public header and will be provided to apps.

  * Remove the namespaces of unused backends (e.g., `namespace cloud::rest`).

  * Update the plugin GUIDs in this file by pasting the code provided by the NVIGI utilities tool:

    ```c++
    // Before
    // constexpr PluginID kId  = { {0x54bbefba, 0x535f, 0x4d77,{0x9c, 0x3f, 0x46, 0x38, 0x39, 0x2d, 0x23, 0xac}}, 0x4b9ee9 };  // {54BBEFBA-535F-4D77-9C3F-4638392D23AC} [nvigi.plugin.gpt.ggml.$backend]
    // After
    constexpr PluginID kId = { 0x576e1145, 0xf790, 0x46b4, { 0xbf, 0x9a, 0xde, 0x88, 0x9, 0x46, 0x3a, 0x15 } };  // {576E1145-F790-46B4-BF9A-DE8809463A15} [nvigi.plugin.mygpt.ggml.$backend]
    ```

* Open `mygpt.cpp`.

  * Include the new headers.

    ```c++
    // Before
    #include "source/plugins/nvigi.gpt/nvigi_gpt.h"
    #include "source/plugins/nvigi.gpt/ggml/gpt.h"
    // After
    #include "source/plugins/nvigi.mygpt/nvigi_mygpt.h"
    #include "source/plugins/nvigi.mygpt/ggml/mygpt.h"
    ```

  * Update function `getFeatureId()` to return the new plugin ID `return plugin::mygpt::ggml::$backend::kId;`.

* (Optional) Rename namespaces and variables in `mygpt.h/cpp`. These source files will not be visible to apps.

### 3. (Optional) Update Your App to Use the New Plugin

If you already have an app using the original GPT plugin "nvigi.plugin.gpt.ggml.cuda", follow these steps to replace it with the new plugin.

* Replace the public header `#include "nvigi_gpt.h"` with `#include "nvigi_mygpt.h"`.
* Replace the plugin ID `nvigi::plugin::gpt::ggml::$backend::kId` with `nvigi::plugin::mygpt::ggml::$backend::kId` when getting the interface.
* Rename the model folder from `nvigi.models\nvigi.plugin.gpt.ggml` to `nvigi.models\nvigi.plugin.mygpt.ggml` so the new plugin can find and load models.
* (Troubleshooting) If the new plugin DLL fails to load, start with checking if you are using the correct set of dependent DLLs, like ggml and llama.cpp and CUDA runtime.

### 4. Customize Your Plugin!

You have completed the setup and can now start modifying the plugin if you have plans in mind. If not, the example below shows how to add a new parameter `grammar` to `GPTSamplerParameters`.

The example demonstrates the steps to add the `grammar` parameter. Note that this parameter may be added in future versions. Similar steps can be applied to other parameters like `GPTRuntimeParameters`. 

* Open `nvigi_mygpt.h` and navigate to `GPTSamplerParameters`, then add the parameter at the end and update the version.

  ```c++
  // Before
  struct alignas(8) GPTSamplerParameters
  {
      NVIGI_UID(UID({ 0xfd183aa9, 0x6e50, 0x4021,{0x9b, 0x0e, 0xa7, 0xae, 0xab, 0x6e, 0xef, 0x49} }), kStructVersion1)
      // ... v1 parameters
      //! v2+ members go here, remember to update the kStructVersionN in the above NVIGI_UID macro!
  };
  // After
  struct alignas(8) GPTSamplerParameters
  {
      NVIGI_UID(UID({ 0xfd183aa9, 0x6e50, 0x4021,{0x9b, 0x0e, 0xa7, 0xae, 0xab, 0x6e, 0xef, 0x49} }), kStructVersion2)
      // ... v1 parameters
      //! v2+ members go here, remember to update the kStructVersionN in the above NVIGI_UID macro!
      const char* grammar{};
  };
  ```

* Open `mygpt.cpp` and navigate to function `ggmlEvaluate()`.

  ```c++
  // Before
  instance->params.sparams.ignore_eos = sampler->ignoreEOS;
  // After
  instance->params.sparams.ignore_eos = sampler->ignoreEOS;
  if (sampler->getVersion() >= 2)
  {
      instance->params.sparams.grammar = sampler->grammar ? sampler->grammar : "";
  }
  ```

* Update your app to set the new parameter.

### 5. Adding New Structures Or Interfaces

If you need to add completely new structure or new interface (structure with functions defining an API) please follow these steps:

* Find the NVIGI utility `nvigi.tool.utils.exe` located in `NVIGI-Core\bin\Release_x64`
* Open terminal and run `nvigi.tool.utils.exe --interface MyData` (make sure to replace `MyData` accordingly)
* Paste the provided code into your `nvigi_mygpt.h` header like for example:

> NOTE: Same command is used for data and interfaces since they are simply typed and versioned structures in NVIGI terminology

```cpp

//! Interface 'MyData'
//!
//! {45DF99CE-F5B8-4D66-90EE-FADEEFBBF713}
struct alignas(8) MyData
{
    MyData() { };
    NVIGI_UID(UID({0x45df99ce, 0xf5b8, 0x4d66,{0x90, 0xee, 0xfa, 0xde, 0xef, 0xbb, 0xf7, 0x13}}), kStructVersion1)

    //! v1 members go here, please do NOT break the C ABI compatibility:

    //! * do not use virtual functions, volatile, STL (e.g. std::vector) or any other C++ high level functionality
    //! * do not use nested structures, always use pointer members
    //! * do not use internal types in _public_ interfaces (like for example 'nvigi::types::vector' etc.)
    //! * do not change or move any existing members once interface has shipped

    //! v2+ members go here, remember to update the kStructVersionN in the above NVIGI_UID macro!
};

NVIGI_VALIDATE_STRUCT(MyData)

```
* Add new data members to your structure
  
### 6. Using New Structures Or Interfaces

#### 6.1 Data Structures

If your new structure contains only data simply chain it to the creation or runtime properties and then use `findStruct` in either `createInstance` or `evaluate` calls.

`host app`
```cpp

// Default GPT creation parameters
GPTCreationParameters creationParams{};
// Fill in the creation params 
MyData myData{};
// Fill in your data
myData.someData = 1;
// Chain it to the creation parameters
if(NVIGI_FAILED(error,creationParams.chain(myData)))
{
  // handle error
}
```

`mygpt.cpp`
```cpp
nvigi::Result ggmlCreateInstance(const nvigi::NVIGIParameter* _params, nvigi::InferenceInstance** _instance)
{
    auto myData = findStruct<MyData>(_params);
    if(myData)
    {
      // Do something with v1 data

      // If you modify your structure after having shipped your app, make sure to bump the version and check
      if(myData->getVersion() >= kStructVersion2)
      {
        // Do something with v2 data
      }
      // And so on ...
    }
    ...
}
```

#### 6.2 Interface Structures

If your new structure contains and API then this new interface must be exported by your modified plugin in order to be used on the host side. Here is an example:

`nvigi_mygpt.h`
```cpp
//! Interface 'IMyInterface'
//!
//! {45DF99CE-F5B8-4D66-90EE-FADEEFBBF713}
struct alignas(8) IMyInterface
{
    IMyInterface() { };
    NVIGI_UID(UID({0x45df99ce, 0xf5b8, 0x4d66,{0x90, 0xee, 0xfa, 0xde, 0xef, 0xbb, 0xf7, 0x13}}), kStructVersion1)

    //! v1 members go here, please do NOT break the C ABI compatibility:

    nvigi::Result (*someFunction)();

    //! * do not use virtual functions, volatile, STL (e.g. std::vector) or any other C++ high level functionality
    //! * do not use nested structures, always use pointer members
    //! * do not use internal types in _public_ interfaces (like for example 'nvigi::types::vector' etc.)
    //! * do not change or move any existing members once interface has shipped

    //! v2+ members go here, remember to update the kStructVersionN in the above NVIGI_UID macro!
};

NVIGI_VALIDATE_STRUCT(IMyInterface)
```

`mygpt.cpp`

```cpp

// Add new API to the context

struct GPTContext
{
    NVIGI_PLUGIN_CONTEXT_CREATE_DESTROY(GPTContext);

    void onCreateContext() {};
    void onDestroyContext() {};
    
    IMyInterface myapi{};

    // other bits
};

namespace mygpt
{

nvigi::Result someFunction()
{
  // Implement your function!  
}

}

// Now export the API
Result nvigiPluginRegister(framework::IFramework* framework)
{
    if (!plugin::internalPluginSetup(framework)) return kResultInvalidState;

    auto& ctx = (*mygpt::getContext());

    ctx.feature = mygpt::getFeatureId(nullptr);

    ctx.myapi.someFunction = mygpt::someFunction;
    
    framework->addInterface(ctx.feature, &ctx.myapi, 0);

    ...
}
```

`host app`
```cpp
nvigi::mygpt::IMyInterface* iMyApi{};
if(NVIGI_FAILED(error,nvigiGetInterface("nvigi::plugin::mygpt::ggml::$backend::kId", &iMyApi)))
{
  // handle error
}

// use your interface v1
iMyApi->someFunction();

// If interface changed and your app can load both old and new plugins always check the version
if(iMyApi->getVersion() >= kStructureVersion2)
{
  // safe to use v2 interface members
  iMyApi->someFunction2();
}

```

