#include <Rcpp.h>
#include <iostream>
#define LLTM_HEADERS_ONLY
#include "lltm/lltm.h"
#define TORCH_IMPL
#define IMPORT_TORCH
#include <torch.h>


// [[Rcpp::export]]
XPtrTorchTensorList lltm_forward (
    XPtrTorchTensor input,
    XPtrTorchTensor weights,
    XPtrTorchTensor bias,
    XPtrTorchTensor old_h,
    XPtrTorchTensor old_cell)
{
  return c_lltm_forward(
    input.get(),
    weights.get(),
    bias.get(),
    old_h.get(),
    old_cell.get()
  );
}

// [[Rcpp::export]]
XPtrTorchTensorList lltm_backward (
    XPtrTorchTensor grad_h,
    XPtrTorchTensor grad_cell,
    XPtrTorchTensor new_cell,
    XPtrTorchTensor input_gate,
    XPtrTorchTensor output_gate,
    XPtrTorchTensor candidate_cell,
    XPtrTorchTensor X,
    XPtrTorchTensor gate_weights,
    XPtrTorchTensor weights)
{
  return c_lltm_backward(
    grad_h.get(),
    grad_cell.get(),
    new_cell.get(),
    input_gate.get(),
    output_gate.get(),
    candidate_cell.get(),
    X.get(),
    gate_weights.get(),
    weights.get()
  );
}

