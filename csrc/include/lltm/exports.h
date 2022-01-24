// Generated by using torchexport::export() -> do not edit by hand
#ifdef _WIN32
#ifndef LLTM_HEADERS_ONLY
#define LLTM_API extern "C" __declspec(dllexport)
#else
#define LLTM_API extern "C" __declspec(dllimport)
#endif
#else
#define LLTM_API extern "C"
#endif

#ifndef LLTM_HANDLE_EXCEPTION
#define LLTM_HANDLE_EXCEPTION                                  \
catch(const std::exception& ex) {                                  \
  p_lltm_last_error = make_raw::string(ex.what());             \
} catch (std::string& ex) {                                        \
  p_lltm_last_error = make_raw::string(ex);                    \
} catch (...) {                                                    \
  p_lltm_last_error = make_raw::string("Unknown error. ");     \
}
#endif

void host_exception_handler ();
extern void* p_lltm_last_error;
LLTM_API void* lltm_last_error ();
LLTM_API void lltm_last_error_clear();

LLTM_API void* _c_lltm_forward (void* input, void* weights, void* bias, void* old_h, void* old_cell);
LLTM_API void* _c_lltm_backward (void* grad_h, void* grad_cell, void* new_cell, void* input_gate, void* output_gate, void* candidate_cell, void* X, void* gate_weights, void* weights);

#ifdef RCPP_VERSION
inline void* c_lltm_forward (void* input, void* weights, void* bias, void* old_h, void* old_cell) {
  auto ret =  _c_lltm_forward(input, weights, bias, old_h, old_cell);
  host_exception_handler();
  return ret;
}
inline void* c_lltm_backward (void* grad_h, void* grad_cell, void* new_cell, void* input_gate, void* output_gate, void* candidate_cell, void* X, void* gate_weights, void* weights) {
  auto ret =  _c_lltm_backward(grad_h, grad_cell, new_cell, input_gate, output_gate, candidate_cell, X, gate_weights, weights);
  host_exception_handler();
  return ret;
}
#endif // RCPP_VERSION