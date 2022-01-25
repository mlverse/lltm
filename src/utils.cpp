#include <Rcpp.h>
#include <iostream>
#define LLTM_HEADERS_ONLY
#include <lltm/lltm.h>
#include <torch.h>

void host_exception_handler ()
{
  if (lltm_last_error())
  {
    auto msg = Rcpp::as<std::string>(torch::string(lltm_last_error()));
    lltm_last_error_clear();
    Rcpp::stop(msg);
  }
}
