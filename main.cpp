#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <tiffio.h>
#include <vector>
#include <algorithm>

using namespace std;
namespace fs = std::filesystem;


int main()
{
    const string input_folder = argv[1];
    const string output_file_path = argv[2];
    int num_classes = 10; 

    cudnnHandle_t handle_ = createCudaHandleAndOutputHWSpecs();
    ofstream output_file(output_file_path);

    for (const auto& entry : fs::directory_iterator(input_folder)) {
        if (entry.path().extension() == ".tiff" || entry.path().extension() == ".tif") {
            auto [input_desc, input_data, n, h, w] = loadImageAndPreprocess(entry.path().c_str());
            float* output_data = runCuDnnModel(handle_, input_desc, input_data, num_classes);
            printClassificationResults(output_file, entry.path().string(), output_data, num_classes);

            cudaFree(input_data);
            cudaFree(output_data);
        }
    }

    cudnnDestroy(handle_);
    std::cout << "Destroyed cuDNN handle." << std::endl;
    output_file.close();

    return 0;
}
