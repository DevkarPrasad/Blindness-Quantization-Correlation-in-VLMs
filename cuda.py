import llama_cpp
print(f"Llama-cpp version: {llama_cpp.__version__}")
# This will print the internal system info. Look for "BLAS = 1" or "CUDA = 1"
print(llama_cpp.llama_print_system_info())