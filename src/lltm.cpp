#include <Rcpp.h>
#define LLTM_HEADERS_ONLY
#include <lltm/lltm.h>
#define TORCH_IMPL
#define IMPORT_TORCH
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

// [[Rcpp::export]]
void lltm_raise_exception ()
{
  raise_exception();
}
