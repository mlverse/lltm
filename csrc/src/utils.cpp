#include <lantern/types.h>
#include <iostream>
#include <vector>
#include "lltm/lltm.h"

void * p_lltm_last_error = NULL;

LLTM_API void* lltm_last_error()
{
  return p_lltm_last_error;
}

LLTM_API void lltm_last_error_clear()
{
  p_lltm_last_error = NULL;
}

