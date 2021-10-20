test_that("multiplication works", {
  x <- torch::torch_tensor(c(1,2,3))
  expect_true(
    torch::torch_allclose(torch::torch_sigmoid(x), run(x))
  )
})
