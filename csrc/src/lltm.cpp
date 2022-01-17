#include <torch/torch.h>
#define LANTERN_TYPES_IMPL // Should be defined only in a single file.
#include <lantern/types.h>
#include <iostream>
#include <vector>
#include "lltm/lltm.h"

std::string *p_lltm_last_error = NULL;

// https://stackoverflow.com/questions/40487123/how-to-wrap-calls-with-try-catch-block
#define HANDLE_LLTM_EXCEPTION(...)                              \
[&]() -> decltype(auto) {                                       \
  try {                                                         \
    return __VA_ARGS__;                                         \
  } catch(const std::exception& ex) {                           \
    p_lltm_last_error = new std::string(ex.what());             \
  } catch (std::string& ex) {                                   \
    p_lltm_last_error = new std::string(ex);                    \
  } catch (...) {                                               \
    p_lltm_last_error = new std::string("Unknown error. ");     \
  }                                                             \
  return (void*)nullptr;                                        \
}()                                                            \

#define HANDLE_EXCEPTION                                       \
catch(const std::exception& ex) {                             \
  p_lltm_last_error = new std::string(ex.what());             \
} catch (std::string& ex) {                                   \
  p_lltm_last_error = new std::string(ex);                    \
} catch (...) {                                               \
  p_lltm_last_error = new std::string("Unknown error. ");     \
}

torch::Tensor d_sigmoid(torch::Tensor z) {
  auto s = torch::sigmoid(z);
  return (1 - s) * s;
}

std::vector<at::Tensor> lltm_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell) {
  auto X = torch::cat({old_h, input}, /*dim=*/1);

  auto gate_weights = torch::addmm(bias, X, weights.transpose(0, 1));
  auto gates = gate_weights.chunk(3, /*dim=*/1);

  auto input_gate = torch::sigmoid(gates[0]);
  auto output_gate = torch::sigmoid(gates[1]);
  auto candidate_cell = torch::elu(gates[2], /*alpha=*/1.0);

  auto new_cell = old_cell + candidate_cell * input_gate;
  auto new_h = torch::tanh(new_cell) * output_gate;

  return {new_h,
          new_cell,
          input_gate,
          output_gate,
          candidate_cell,
          X,
          gate_weights};
}

// tanh'(z) = 1 - tanh^2(z)
torch::Tensor d_tanh(torch::Tensor z) {
  return 1 - z.tanh().pow(2);
}

// elu'(z) = relu'(z) + { alpha * exp(z) if (alpha * (exp(z) - 1)) < 0, else 0}
torch::Tensor d_elu(torch::Tensor z, torch::Scalar alpha = 1.0) {
  auto e = z.exp();
  auto mask = (alpha * (e - 1)) < 0;
  return (z > 0).type_as(z) + mask.type_as(z) * (alpha * e);
}

std::vector<torch::Tensor> lltm_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell,
    torch::Tensor new_cell,
    torch::Tensor input_gate,
    torch::Tensor output_gate,
    torch::Tensor candidate_cell,
    torch::Tensor X,
    torch::Tensor gate_weights,
    torch::Tensor weights) {
  auto d_output_gate = torch::tanh(new_cell) * grad_h;
  auto d_tanh_new_cell = output_gate * grad_h;
  auto d_new_cell = d_tanh(new_cell) * d_tanh_new_cell + grad_cell;

  auto d_old_cell = d_new_cell;
  auto d_candidate_cell = input_gate * d_new_cell;
  auto d_input_gate = candidate_cell * d_new_cell;

  auto gates = gate_weights.chunk(3, /*dim=*/1);
  d_input_gate *= d_sigmoid(gates[0]);
  d_output_gate *= d_sigmoid(gates[1]);
  d_candidate_cell *= d_elu(gates[2]);

  auto d_gates =
    torch::cat({d_input_gate, d_output_gate, d_candidate_cell}, /*dim=*/1);

  auto d_weights = d_gates.t().mm(X);
  auto d_bias = d_gates.sum(/*dim=*/0, /*keepdim=*/true);

  auto d_X = d_gates.mm(weights);
  const auto state_size = grad_h.size(1);
  auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
  auto d_input = d_X.slice(/*dim=*/1, state_size);

  return {d_old_h, d_input, d_weights, d_bias, d_old_cell};
}

void* _c_lltm_forward (void* input,
                       void* weights,
                       void* bias,
                       void* old_h,
                       void* old_cell) noexcept {
  try {
    return make_raw::TensorList(lltm_forward(
        from_raw::Tensor(input),
        from_raw::Tensor(weights),
        from_raw::Tensor(bias),
        from_raw::Tensor(old_h),
        from_raw::Tensor(old_cell)
    ));
  } HANDLE_EXCEPTION
  return (void*) nullptr;
}

LLTM_API void* c_lltm_backward(
    void* grad_h,
    void* grad_cell,
    void* new_cell,
    void* input_gate,
    void* output_gate,
    void* candidate_cell,
    void* X,
    void* gate_weights,
    void* weights) {

  auto result = lltm_backward(
    from_raw::Tensor(grad_h),
    from_raw::Tensor(grad_cell),
    from_raw::Tensor(new_cell),
    from_raw::Tensor(input_gate),
    from_raw::Tensor(output_gate),
    from_raw::Tensor(candidate_cell),
    from_raw::Tensor(X),
    from_raw::Tensor(gate_weights),
    from_raw::Tensor(weights)
  );

  return make_raw::TensorList(result);
}
