// Generated by using torchexport::export() -> do not edit by hand
#include "lltm/exports.h"
#include <lantern/types.h>
void * p_lltm_last_error = NULL;

LLTM_API void* lltm_last_error()
{
  return p_lltm_last_error;
}

LLTM_API void lltm_last_error_clear()
{
  p_lltm_last_error = NULL;
}

void* c_lltm_forward (void* input, void* weights, void* bias, void* old_h, void* old_cell);
LLTM_API void* _c_lltm_forward (void* input, void* weights, void* bias, void* old_h, void* old_cell) {
  try {
    return  c_lltm_forward(input, weights, bias, old_h, old_cell);
  } LLTM_HANDLE_EXCEPTION
  return ( void* ) NULL;
}
void* c_lltm_backward (void* grad_h, void* grad_cell, void* new_cell, void* input_gate, void* output_gate, void* candidate_cell, void* X, void* gate_weights, void* weights);
LLTM_API void* _c_lltm_backward (void* grad_h, void* grad_cell, void* new_cell, void* input_gate, void* output_gate, void* candidate_cell, void* X, void* gate_weights, void* weights) {
  try {
    return  c_lltm_backward(grad_h, grad_cell, new_cell, input_gate, output_gate, candidate_cell, X, gate_weights, weights);
  } LLTM_HANDLE_EXCEPTION
  return ( void* ) NULL;
}