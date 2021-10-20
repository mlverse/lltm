#include <torch/torch.h>
#include <iostream>
#include "lltm/lltm.h"

template <class T>
class LanternObject
{
private:
  T _object;

public:
  LanternObject(T object) : _object(std::forward<T>(object))
  {
  }

  LanternObject()
  {
  }

  T &get()
  {
    return _object;
  }
};

LLTM_API void* d_sigmoid(void* x)
{
  torch::Tensor s = reinterpret_cast<LanternObject<torch::Tensor>*>(x)->get();
  auto out = torch::sigmoid(s);
  return (void*) new LanternObject<torch::Tensor>(out);
}
