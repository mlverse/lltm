#ifdef _WIN32
#ifndef LLTM_HEADERS_ONLY
#define LLTM_API extern "C" __declspec(dllexport)
#else
#define LLTM_API extern "C"
#endif
#else
#define LLTM_API extern "C"
#endif

LLTM_API int d_sigmoid (int z);


