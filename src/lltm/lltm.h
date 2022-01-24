#ifdef _WIN32
#ifndef LLTM_HEADERS_ONLY
#define LLTM_API extern "C" __declspec(dllexport)
#else
#define LLTM_API extern "C" __declspec(dllimport)
#endif
#else
#define LLTM_API extern "C"
#endif

void host_exception_handler ();
extern void* p_lltm_last_error;

// Exceptions that happen inside functions from the C API must be handled back
// in the host side (eg. in Rcpp) in so we can correctly return the exception
// to R.
// This function returns a modified function that calls a host handler whenever
// it's out of scope. Currently, the way we propose handling the exceptions is
// to use the host_exception_handler() to check a global variable that is either
// NULL if no exception ocurred or a string if an exception has been raised while
// executing the C function.
// Use this function by wrapping your C function into it.
// This adds C++14 dependency. If the C++ dependency is really undesirable you
// can be more verbose declaring a function with the same signature that calls
// the C interface, and calls host_exception_handler() before returning. Note:
// `host_exception_handler()` might raise exceptions so it's not a good idea to
// call it in ScopeGuards or similar strategies.
// Also this function only works if the wrapped function doesn't return void.
inline auto handle_exceptions = [](const auto& f) {
  return [&](auto ... params) {
    auto ret = f(params...);
    host_exception_handler();
    return ret;
  };
};

LLTM_API void* lltm_last_error ();
LLTM_API void lltm_last_error_clear();

LLTM_API void* _c_lltm_forward (void* input, void* weights, void* bias, void* old_h,
                                void* old_cell) noexcept;
static auto c_lltm_forward = handle_exceptions(_c_lltm_forward);

LLTM_API void* c_lltm_backward (void* grad_h, void* grad_cell, void* new_cell,
                                void* input_gate, void* output_gate, void* candidate_cell,
                                void* X, void* gate_weights, void* weights);

LLTM_API int _raise_exception ();
static auto raise_exception = handle_exceptions(_raise_exception);



