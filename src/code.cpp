#include <Rcpp.h>
#include <iostream>
#define LLTM_HEADERS_ONLY
#include "lltm/lltm.h"


// [[Rcpp::export]]
int run () {
  return d_sigmoid(3);
}

using namespace Rcpp;
