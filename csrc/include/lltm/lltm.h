#ifdef _WIN32
#define LLTM_API extern "C" __declspec(dllexport)
#else
#define LLTM_API extern "C"
#endif

LLTM_API int d_sigmoid (int z);


