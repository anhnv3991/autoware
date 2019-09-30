#include "ndt_gpu/MatrixHost.h"
#include "ndt_gpu/debug.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

namespace gpu {

template <typename Scalar, int Rows, int Cols>
MatrixHost<Scalar, Rows, Cols>::MatrixHost()
{
	is_copied_ = false;
	buffer_ = NULL;
	offset_ = 0;
}

template <typename Scalar, int Rows, int Cols>
MatrixHost<Scalar, Rows, Cols>::MatrixHost(int offset, Scalar *buffer) :
Matrix<Scalar, Rows, Cols>(offset, buffer)
{}

template <typename Scalar, int Rows, int Cols>
MatrixHost<Scalar, Rows, Cols>::MatrixHost(const MatrixHost<Scalar, Rows, Cols>& other) {
	if (rows_ > 0 && cols_ > 0) {
		offset_ = other.offset_;
		is_copied_ = true;
		rows_ = other.rows_;
		cols_ = other.cols_;
		buffer_ = other.buffer_;
	}
}

template <typename Scalar, int Rows, int Cols>
__global__ void copyMatrixDevToDev(MatrixDevice<Scalar, Rows, Cols> input, MatrixDevice<Scalar, Rows, Cols> output) {
	int row = threadIdx.x;
	int col = threadIdx.y;

	if (row < input.rows() && col < input.cols())
		output(row, col) = input(row, col);
}

template <typename Scalar, int Rows, int Cols>
bool MatrixHost<Scalar, Rows, Cols>::moveToGpu(MatrixDevice<Scalar, Rows, Cols> output) {
	if (rows_ != output.rows() || cols_ != output.cols())
		return false;

	Scalar *tmp;

	checkCudaErrors(cudaMalloc(&tmp, sizeof(Scalar) * rows_ * cols_ * offset_));
	checkCudaErrors(cudaMemcpy(tmp, buffer_, sizeof(Scalar) * rows_ * cols_ * offset_, cudaMemcpyHostToDevice));

	MatrixDevice<Scalar, Rows, Cols> tmp_output(rows_, cols_, offset_, tmp);

	dim3 block_x(rows_, cols_, 1);
	dim3 grid_x(1, 1, 1);

	copyMatrixDevToDev<Scalar, Rows, Cols><<<grid_x, block_x>>>(tmp_output, output);
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(tmp));

	return true;
}

template <typename Scalar, int Rows, int Cols>
bool MatrixHost<Scalar, Rows, Cols>::moveToHost(const MatrixDevice<Scalar, Rows, Cols> input) {
	if (rows_ != input.rows() || cols_ != input.cols())
		return false;

	Scalar *tmp;

	checkCudaErrors(cudaMalloc(&tmp, sizeof(Scalar) * rows_ * cols_ * offset_));

	MatrixDevice<Scalar, Rows, Cols> tmp_output(rows_, cols_, offset_, tmp);

	dim3 block_x(rows_, cols_, 1);
	dim3 grid_x(1, 1, 1);

	copyMatrixDevToDev<Scalar, Rows, Cols><<<grid_x, block_x>>>(input, tmp_output);
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(buffer_, tmp, sizeof(Scalar) * rows_ * cols_ * offset_, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(tmp));

	return true;

}

template <typename Scalar, int Rows, int Cols>
MatrixHost<Scalar, Rows, Cols> &MatrixHost<Scalar, Rows, Cols>::operator=(const MatrixHost<Scalar, Rows, Cols> &other)
{
	buffer_ = other.buffer_;
	offset_ = other.offset_;
	is_copied_ = true;
	rows_ = other.rows_;
	cols_ = other.cols_;

	return *this;
}

template <typename Scalar, int Rows, int Cols>
MatrixHost<Scalar, Rows, Cols> &MatrixHost<Scalar, Rows, Cols>::operator=(MatrixHost<Scalar, Rows, Cols> &&other)
{
	if (!is_copied_ && buffer_ != NULL) {
		free(buffer_);
		is_copied_ = false;
	}

	offset_ = other.offset_;
	is_copied_ = false;
	buffer_ = other.buffer_;
	rows_ = other.rows_;
	cols_ = other.cols_;

	other.is_copied_ = true;
	other.buffer_ = NULL;

	return *this;
}


template <typename Scalar, int Rows, int Cols>
void MatrixHost<Scalar, Rows, Cols>::debug()
{
	std::cout << *this;
}

template <typename Scalar, int Rows, int Cols>
std::ostream &operator<<(std::ostream &os, const MatrixHost<Scalar, Rows, Cols> &value)
{
	for (int i = 0; i < value.rows(); i++) {
		for (int j = 0; j < value.cols(); j++) {
			os << value.at(i, j) << " ";
		}

		os << std::endl;
	}

	os << std::endl;

	return os;
}

template <typename Scalar, int Rows, int Cols>
MatrixHost<Scalar, Rows, Cols>::~MatrixHost()
{
	if (!is_copied_ && buffer_ != NULL)
		free(buffer_);
}

template class MatrixHost<float>;
template class MatrixHost<double>;

}
