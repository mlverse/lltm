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

inst_path <- function() {
  install_path <- Sys.getenv("LLTM_HOME")
  if (nzchar(install_path)) return(install_path)

  if (.Platform$OS.type == "unix") {
    paste0("csrc/build/")
  } else {
    paste0("csrc/build/Debug/")
  }
}

lib_path <- function() {
  install_path <- inst_path()

  if (.Platform$OS.type == "unix") {
    file.path(install_path, paste0("liblltm", lib_ext()))
  } else {
    file.path(install_path, paste0("lltm", lib_ext()))
  }
}

lib_ext <- function() {
  if (grepl("darwin", version$os))
    ".dylib"
  else if (grepl("linux", version$os))
    ".so"
  else
    ".dll"
}


