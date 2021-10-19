#include <torch/torch.h>
#include <iostream>
#include "lltm/lltm.h"


LLTM_API int d_sigmoid(int a)
{
  auto z = torch::randn({a, a});
  auto s = torch::sigmoid(z);
  std::cout << s << std::endl;
  return 1;
}
