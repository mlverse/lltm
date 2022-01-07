#include <Rcpp.h>
#include <iostream>
#define LLTM_HEADERS_ONLY
#include "lltm/lltm.h"
#define TORCH_IMPL
#define IMPORT_TORCH
#include <torch.h>


// [[Rcpp::export]]
torch::TensorList lltm_forward (
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell)
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
torch::TensorList lltm_backward (
    torch::Tensor grad_h,
    torch::Tensor grad_cell,
    torch::Tensor new_cell,
    torch::Tensor input_gate,
    torch::Tensor output_gate,
    torch::Tensor candidate_cell,
    torch::Tensor X,
    torch::Tensor gate_weights,
    torch::Tensor weights)
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
