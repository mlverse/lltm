#include <lltm/exports.h>

auto handle_exceptions = [](const auto& f) {
  return [&](auto ... params) {
    auto ret = f(params...);
    host_exception_handler();
    return ret;
  };
};

LLTM_API int _raise_exception ();
static auto raise_exception = handle_exceptions(_raise_exception);



