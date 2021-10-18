## usethis namespace: start
#' @importFrom Rcpp sourceCpp
## usethis namespace: end
NULL

.onLoad <- function(lib, pkg) {
  if (torch::torch_is_installed()) {
    dyn.load(lib_path(), local = FALSE)

    # when using devtools::load_all() the library might be available in
    # `lib/pkg/src`
    pkgload <- file.path(lib, pkg, "src", paste0(pkg, .Platform$dynlib.ext))
    if (file.exists(pkgload))
      dyn.load(pkgload)
    else
      library.dynam(pkg, pkg, lib)
  }
}

lib_path <- function() {
  if (.Platform$OS.type == "unix") {
    paste0("csrc/build/liblltm", lib_ext())
  } else {
    paste0("csrc/build/Release/liblltm", lib_ext())
  }
}

lib_ext <- function() {
  if (grepl(version$os, "darwin"))
    ".dylib"
  else if (grepl(version$os, "linux"))
    ".so"
  else
    ".dll"
}


