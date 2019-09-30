#include "ndt_gpu/MatrixDevice.h"
#include "ndt_gpu/debug.h"

namespace gpu {
template <typename Scalar, int Rows, int Cols>
MatrixDevice<Scalar, Rows, Cols>::MatrixDevice()
{
	is_copied_ = false;

	offset_ = 1;

	checkCudaErrors(cudaMalloc(&buffer_, sizeof(Scalar) * Rows * Cols * offset_));
	checkCudaErrors(cudaMemset(buffer_, 0, sizeof(Scalar) * Rows * Cols * offset_));
	checkCudaErrors(cudaDeviceSynchronize());
}


template <typename Scalar, int Rows, int Cols>
void MatrixDevice<Scalar, Rows, Cols>::free()
{
	if (!is_copied_ && buffer_ != nullptr) {
		checkCudaErrors(cudaFree(buffer_));
		buffer_ = nullptr;
	}
}

template class MatrixDevice<float, 3, 1>;
template class MatrixDevice<double, 3, 1>;
template class MatrixDevice<double, 3, 3>;

}
