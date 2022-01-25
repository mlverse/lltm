#include <lltm/exports.h>

LLTM_API int _raise_exception ();
inline int raise_exception () {
  _raise_exception();
  host_exception_handler();
  return 1;
}




