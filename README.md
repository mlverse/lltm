
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
library(lltm)
## basic example code
```
