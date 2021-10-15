#include <Rcpp.h>
#include <iostream>
#include "../csrc/include/lltm/lltm.h"

// [[Rcpp::export]]
int run () {
  return d_sigmoid(3);
}

using namespace Rcpp;
