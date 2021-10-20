#ifdef _WIN32
#ifndef LLTM_HEADERS_ONLY
#define LLTM_API extern "C" __declspec(dllexport)
#else
#define LLTM_API extern "C" __declspec(dllimport)
#endif
#else
#define LLTM_API extern "C"
#endif

LLTM_API void* d_sigmoid(void* x);


