
<!-- README.md is generated from README.Rmd. Please edit that file -->

# lltm

<!-- badges: start -->
<!-- badges: end -->

The goal of lltm is to be a minimal implementation of an extension for
[torch](https://github.com/mlverse/torch) that interfaces with the
underlying C++ interface, called LibTorch.

In this pakage we provide an implementation of a new recurrent unit that
is similar to a LSTM but it lacks a *forget gate* and uses an
*Exponential Linear Unit* (ELU) as its internal activation function.
Because this unit never forgets, we’ll call it LLTM, or
**Long-Long-Term-Memory unit**.

The example implemented here is a port of the official PyTorch
[tutorial](https://pytorch.org/tutorials/advanced/cpp_extension.html) on
custom C++ and CUDA extensions.

## High-Level overview

Writing C++ extensions for torch requires us to coordinate the
communication between multiple agents in the torch ecossytem. The
following diagram is a high-level overview on how they communicate in
this package.

On the torch package side the agents that appear are:

-   **LibTorch**: The PyTorch’s C++ interface. This is the library
    implementing all the heavy computations and the data structures like
    tensors.
-   **Lantern**: Is a C wrapper for LibTorch and is a part of the torch
    for R project. We had to develop Lantern because on Windows LibTorch
    can only be  
    compiled with the MSVC compiler while R is compiled with MinGW.
    Because of the different compilers, only C interfaces (not C++) are
    compatible.
-   **torchpkg.so**: This is how we are referring to the C++ library,
    implemented with Rcpp that allows the R API to make calls to Lantern
    functions. Another important feature it provides is custom Rcpp
    types that allows users to easily manage memory life time of objects
    returned by Lantern.

In the extension side the actors are:

-   **csrc**: What we are calling `csrc` here is the equivalent to
    Lantern in the torch project. It’s a C interface for calling
    functions from LibTorch that implement the desidered extension
    functionality. The library produced here must also be compiled with
    MSVC on Windows thus the C interface is required.
-   **lltm.so**: This is the C++ library implemented using Rcpp that
    allows the R API to call the `csrc` functionality. Here, in general,
    we want to use the `torchpkg.so` features to manage memory instead
    of re-implementing that functionality.

[![](man/figures/high-level.png)](https://excalidraw.com/#json=6114208240369664,J9vJ8KK7VOBqgn7Nex5Huw)

## Project structure

-   **csrc**: The directory containing library that will call efficient
    LibTorch code. See the section `csrc` for details.
-   **src**: Rcpp code that interfaces the `csrc` library and exports
    functionality to the R API.
-   **R/package.R**: Definitions for correctly downloading pre-built
    binaries, and dynamically loading the `csrc` library as well as the
    C++ library.

### csrc

> Implementing the operators and their C wrappers.

-   **CMakeLists.txt**: The first important file that you should get
    familiar with in this directory is the
    [CMakeLists.txt](https://github.com/mlverse/lltm/blob/main/csrc/CMakeLists.txt)
    file. This is the [CMake](https://cmake.org/) configuration file
    defining how the project must be compiled and its dependencies. You
    can refer to comments in the
    [file](https://github.com/mlverse/lltm/blob/main/csrc/CMakeLists.txt)
    for almost line by line explanation of definitions.

-   **csrc/src/lltm.cpp**: In this file we define the LibTorch
    implementation of the operations we want to export as well as their
    C signature wrappers. For example we define the `lltm_forward`
    implementation with: (For details on the `lltm_forward`
    implementation refer to the [official
    guide](https://pytorch.org/tutorials/advanced/cpp_extension.html).)

    ``` cpp
    std::vector<at::Tensor> lltm_forward(
      torch::Tensor input,
      torch::Tensor weights,
      torch::Tensor bias,
      torch::Tensor old_h,
      torch::Tensor old_cell) {
        auto X = torch::cat({old_h, input}, /*dim=*/1);

        auto gate_weights = torch::addmm(bias, X, weights.transpose(0, 1));
        auto gates = gate_weights.chunk(3, /*dim=*/1);

        auto input_gate = torch::sigmoid(gates[0]);
        auto output_gate = torch::sigmoid(gates[1]);
        auto candidate_cell = torch::elu(gates[2], /*alpha=*/1.0);

        auto new_cell = old_cell + candidate_cell * input_gate;
        auto new_h = torch::tanh(new_cell) * output_gate;

        return {new_h,
                new_cell,
                input_gate,
                output_gate,
                candidate_cell,
                X,
                gate_weights};
    }
    ```

    We can then write a wrapper for that function with a C signature.
    The C signature takes all arguments as C types. In order to pass
    torch Tensors we need to cast them to `void *` pointers and
    *reinterpret* than before calling the C++ implementation. The return
    value of the C wrapper must also have a C type. In order to return
    Tensors we also cast them to `void*` pointers, making sure that what
    they point won’t be deleted after the function goes out of scope.

    ``` cpp
    LLTM_API void* c_lltm_forward (void* input,
                                   void* weights,
                                   void* bias,
                                   void* old_h,
                                   void* old_cell) {

      std::vector<torch::Tensor> output = lltm_forward(
        reinterpret_cast<LanternObject<torch::Tensor>*>(input)->get(),
        reinterpret_cast<LanternObject<torch::Tensor>*>(weights)->get(),
        reinterpret_cast<LanternObject<torch::Tensor>*>(bias)->get(),
        reinterpret_cast<LanternObject<torch::Tensor>*>(old_h)->get(),
        reinterpret_cast<LanternObject<torch::Tensor>*>(old_cell)->get()
      );

      return (void*) new std::vector<torch::Tensor>(output);
    }
    ```

    We repeat this for the `lltm_backward` function.

-   **csrc/src/lltm.def**: In this file we list the functions from
    `lltm.cpp` that we want to export. This is only required for
    Windows, but it’s a good practice to keep it up to date. See more
    information on Module definition files in this
    [link](https://docs.microsoft.com/en-us/cpp/build/reference/module-definition-dot-def-files?view=msvc-160)

    For example, the current definition is:

        LIBRARY LLTM
        EXPORTS
          c_lltm_forward
          c_lltm_backward

-   **csrc/include/lltm/lltm.h**: We also want to keep an up-to-date
    header file with the signatures of all functions that we will want
    to call from the R wrapper. This file will be *included* in the Rcpp
    code.

    For example, for the `lltm_forward` function we would add something
    like this:

    ``` cpp
    LLTM_API void* c_lltm_forward (void* input, void* weights, 
                                   void* bias, void* old_h,
                                   void* old_cell);
    ```

## Installation

~~You can install the released version of lltm from
[CRAN](https://CRAN.R-project.org) with:~~

``` r
install.packages("lltm")
```

And the development version from [GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("mlverse/lltm")
```

## Example

This is a basic example which shows you how to solve a common problem:

``` r
## library(lltm)
## basic example code
```
