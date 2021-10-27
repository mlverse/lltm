#include <Rcpp.h>
#include <iostream>
#define LLTM_HEADERS_ONLY
#include "lltm/lltm.h"
#define TORCH_IMPL
#define IMPORT_TORCH
#include <torch.h>


// [[Rcpp::export]]
XPtrTorchTensor run (XPtrTorchTensor x) {
  return d_sigmoid(x.get());
}







