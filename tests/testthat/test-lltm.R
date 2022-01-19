test_that("multiplication works", {
  batch_size = 16
  input_features = 32
  state_size = 128

  X = torch::torch_randn(batch_size, input_features)
  h = torch::torch_randn(batch_size, state_size)
  C = torch::torch_randn(batch_size, state_size)

  rnn = nn_lltm(input_features, state_size)



  out = rnn(X, list(h, C))
  l <- out[[1]]$sum() + out[[2]]$sum()
  l$backward()

  expect_equal(rnn$weights$grad$shape, c(384, 160))
  expect_equal(rnn$bias$grad$shape, c(384))
})

test_that("raise exceptions", {

  expect_error(lltm_raise_exception(), "Error from LLTM")

})
