lltm_function <- torch::autograd_function(
  forward = function(ctx, input, weights, bias, old_h, old_cell) {
    outputs <- rcpp_lltm_forward(input, weights, bias, old_h, old_cell)
    names(outputs) <- c("new_h", "new_cell", "input_gate", "output_gate",
                        "candidate_cell", "X", "gate_weights")

    variables <- append(outputs, list(weights = weights))
    ctx$save_for_backward(!!!variables)

    outputs[c("new_h", "new_cell")]
  },
  backward = function(ctx, grad_h, grad_cell) {
    outputs <- rcpp_lltm_backward(
      grad_h = grad_h$contiguous(),
      grad_cell = grad_cell$contiguous(),
      new_cell = ctx$saved_variables$new_cell,
      input_gate = ctx$saved_variables$input_gate,
      output_gate = ctx$saved_variables$output_gate,
      candidate_cell = ctx$saved_variables$candidate_cell,
      X = ctx$saved_variables$X,
      gate_weights = ctx$saved_variables$gate_weights,
      weights = ctx$saved_variables$weights
    )

    names(outputs) <- c("old_h", "input", "weights", "bias", "old_cell")
    outputs
  }
)

nn_lltm <- torch::nn_module(
  initialize = function(input_features, state_size) {
    self$input_features <- input_features
    self$state_size <- state_size
    self$weights <- torch::nn_parameter(
      torch::torch_empty(3 * state_size, input_features + state_size))
    self$bias <- torch::nn_parameter(torch::torch_empty(3 * state_size))
    self$reset_parameters()
  },
  reset_parameters = function() {
    stdv = 1.0 / sqrt(self$state_size)
    lapply(self$parameters, function(x) {
        torch::nn_init_uniform_(x, a = -stdv, b = stdv)
    })
  },
  forward = function(input, state) {
    lltm_function(input, self$weights, self$bias, state[[1]], state[[2]])
  }
)
