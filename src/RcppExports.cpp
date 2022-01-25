// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include "lltm_types.h"
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// rcpp_lltm_forward
torch::TensorList rcpp_lltm_forward(torch::Tensor input, torch::Tensor weights, torch::Tensor bias, torch::Tensor old_h, torch::Tensor old_cell);
RcppExport SEXP _lltm_rcpp_lltm_forward(SEXP inputSEXP, SEXP weightsSEXP, SEXP biasSEXP, SEXP old_hSEXP, SEXP old_cellSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< torch::Tensor >::type input(inputSEXP);
    Rcpp::traits::input_parameter< torch::Tensor >::type weights(weightsSEXP);
    Rcpp::traits::input_parameter< torch::Tensor >::type bias(biasSEXP);
    Rcpp::traits::input_parameter< torch::Tensor >::type old_h(old_hSEXP);
    Rcpp::traits::input_parameter< torch::Tensor >::type old_cell(old_cellSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_lltm_forward(input, weights, bias, old_h, old_cell));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_lltm_backward
torch::TensorList rcpp_lltm_backward(torch::Tensor grad_h, torch::Tensor grad_cell, torch::Tensor new_cell, torch::Tensor input_gate, torch::Tensor output_gate, torch::Tensor candidate_cell, torch::Tensor X, torch::Tensor gate_weights, torch::Tensor weights);
RcppExport SEXP _lltm_rcpp_lltm_backward(SEXP grad_hSEXP, SEXP grad_cellSEXP, SEXP new_cellSEXP, SEXP input_gateSEXP, SEXP output_gateSEXP, SEXP candidate_cellSEXP, SEXP XSEXP, SEXP gate_weightsSEXP, SEXP weightsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< torch::Tensor >::type grad_h(grad_hSEXP);
    Rcpp::traits::input_parameter< torch::Tensor >::type grad_cell(grad_cellSEXP);
    Rcpp::traits::input_parameter< torch::Tensor >::type new_cell(new_cellSEXP);
    Rcpp::traits::input_parameter< torch::Tensor >::type input_gate(input_gateSEXP);
    Rcpp::traits::input_parameter< torch::Tensor >::type output_gate(output_gateSEXP);
    Rcpp::traits::input_parameter< torch::Tensor >::type candidate_cell(candidate_cellSEXP);
    Rcpp::traits::input_parameter< torch::Tensor >::type X(XSEXP);
    Rcpp::traits::input_parameter< torch::Tensor >::type gate_weights(gate_weightsSEXP);
    Rcpp::traits::input_parameter< torch::Tensor >::type weights(weightsSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpp_lltm_backward(grad_h, grad_cell, new_cell, input_gate, output_gate, candidate_cell, X, gate_weights, weights));
    return rcpp_result_gen;
END_RCPP
}
// lltm_raise_exception
void lltm_raise_exception();
RcppExport SEXP _lltm_lltm_raise_exception() {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    lltm_raise_exception();
    return R_NilValue;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_lltm_rcpp_lltm_forward", (DL_FUNC) &_lltm_rcpp_lltm_forward, 5},
    {"_lltm_rcpp_lltm_backward", (DL_FUNC) &_lltm_rcpp_lltm_backward, 9},
    {"_lltm_lltm_raise_exception", (DL_FUNC) &_lltm_lltm_raise_exception, 0},
    {NULL, NULL, 0}
};

RcppExport void R_init_lltm(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
