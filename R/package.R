## usethis namespace: start
#' @importFrom Rcpp sourceCpp
## usethis namespace: end
NULL

.onLoad <- function(lib, pkg) {
  if (torch::torch_is_installed()) {
    dyn.load("csrc/build/liblltm.dylib", local = FALSE)

    # when using devtools::load_all() the library might be available in
    # `lib/pkg/src`
    pkgload <- file.path(lib, pkg, "src", paste0(pkg, .Platform$dynlib.ext))
    if (file.exists(pkgload))
      dyn.load(pkgload)
    else
      library.dynam(pkg, pkg, lib)
  }
}
