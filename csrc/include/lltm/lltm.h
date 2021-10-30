#ifdef _WIN32
#ifndef LLTM_HEADERS_ONLY
#define LLTM_API extern "C" __declspec(dllexport)
#else
#define LLTM_API extern "C" __declspec(dllimport)
#endif
#else
#define LLTM_API extern "C"
#endif

LLTM_API void* c_lltm_forward (void* input, void* weights, void* bias, void* old_h,
                               void* old_cell);
LLTM_API void* c_lltm_backward(void* grad_h, void* grad_cell, void* new_cell,
                               void* input_gate, void* output_gate, void* candidate_cell,
                               void* X, void* gate_weights, void* weights);



