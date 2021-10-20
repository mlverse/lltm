#include <Rcpp.h>
#include <iostream>
#define LLTM_HEADERS_ONLY
#include "lltm/lltm.h"
#include <torch.h>

// [[Rcpp::export]]
XPtrTorchTensor run (XPtrTorchTensor x) {
  return d_sigmoid(x.get());
}

