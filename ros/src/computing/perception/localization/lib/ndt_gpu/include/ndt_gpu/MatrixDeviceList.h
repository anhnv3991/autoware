#ifndef MAT_DEV_LIST_H_
#define MAT_DEV_LIST_H_

#include <cuda.h>
#include "debug.h"
#include <iostream>
#include <memory>

namespace gpu {
template <typename Scalar>
class MatrixDeviceList {
public:
	MatrixDeviceList();

	MatrixDeviceList(int rows, int cols, int mat_num);

	MatrixDeviceList(const MatrixDeviceList<Scalar> &other);

	MatrixDeviceList<Scalar> &operator=(const MatrixDeviceList<Scalar> &other);

	MatrixDeviceList(MatrixDeviceList<Scalar> &&other);

	MatrixDeviceList<Scalar> &operator=(MatrixDeviceList<Scalar> &&other);

	bool copy_from(const MatrixDeviceList<Scalar> &other);

	// Return the address of the first element at (row, col)
	CUDAH Scalar *operator()(int row, int col);

	// Return the reference to the element at (mat_id, row, col)
	CUDAH Scalar& operator()(int mat_id, int row, int col);

	CUDAH MatrixDevice<Scalar> operator()(int mat_id);

	int size() {
		return mat_num_;
	}

	int rows() {
		return rows_;
	}

	int cols() {
		return cols_;
	}

	void free();
private:
	Scalar *buffer_;
	int mat_num_;
	bool is_copied_;
	int rows_, cols_;
};

template <typename Scalar>
MatrixDeviceList<Scalar>::MatrixDeviceList()
{
	buffer_ = NULL;
	mat_num_ = 0;
	is_copied_ = false;
	rows_ = cols_ = 0;
}

template <typename Scalar>
MatrixDeviceList<Scalar>::MatrixDeviceList(int rows, int cols, int mat_num)
{
	buffer_ = NULL;
	mat_num_ = mat_num;
	is_copied_ = false;
	rows_ = rows;
	cols_ = cols;

	checkCudaErrors(cudaMalloc(&buffer_, sizeof(Scalar) * rows_ * cols_ * mat_num_));
	checkCudaErrors(cudaMemset(buffer_, 0, sizeof(Scalar) * rows_ * cols_ * mat_num_));
	checkCudaErrors(cudaDeviceSynchronize());
}

template <typename Scalar>
MatrixDeviceList<Scalar>::MatrixDeviceList(const MatrixDeviceList<Scalar> &other)
{
	buffer_ = other.buffer_;
	mat_num_ = other.mat_num_;
	is_copied_ = true;
	rows_ = other.rows_;
	cols_ = other.cols_;
}

template <typename Scalar>
MatrixDeviceList<Scalar> &MatrixDeviceList<Scalar>::operator=(const MatrixDeviceList<Scalar> &other)
{
	buffer_ = other.buffer_;
	mat_num_ = other.mat_num_;
	is_copied_ = true;
	rows_ = other.rows_;
	cols_ = other.cols_;

	return *this;
}


template <typename Scalar>
MatrixDeviceList<Scalar>::MatrixDeviceList(MatrixDeviceList<Scalar> &&other)
{
	buffer_ = other.buffer_;
	mat_num_ = other.mat_num_;
	is_copied_ = false;
	rows_ = other.rows_;
	cols_ = other.cols_;

	other.buffer_ = NULL;
	other.is_copied_ = true;
}

template <typename Scalar>
MatrixDeviceList<Scalar> &MatrixDeviceList<Scalar>::operator=(MatrixDeviceList<Scalar> &&other)
{
	buffer_ = other.buffer_;
	mat_num_ = other.mat_num_;
	is_copied_ = false;
	rows_ = other.rows_;
	cols_ = other.cols_;

	other.buffer_ = NULL;
	other.is_copied_ = true;

	return *this;
}

template <typename Scalar>
bool MatrixDeviceList<Scalar>::copy_from(const MatrixDeviceList<Scalar> &other)
{
	if (mat_num_ != other.mat_num_ || rows_ != other.rows_ || cols_ != other.cols_)
		return false;

	checkCudaErrors(cudaMemcpy(buffer_, other.buffer_, sizeof(Scalar) * rows_ * cols_ * mat_num_, cudaMemcpyDeviceToDevice));

	return true;
}

template <typename Scalar>
CUDAH Scalar *MatrixDeviceList<Scalar>::operator()(int row, int col)
{
	if (row >= rows_ || col >= cols_ || row < 0 || col < 0)
		return NULL;

	if (mat_num_ == 0 || buffer_ == NULL)
		return NULL;

	return (buffer_ + row * cols_ + col);
}

template <typename Scalar>
CUDAH Scalar &MatrixDeviceList<Scalar>::operator()(int mat_id, int row, int col)
{
	return buffer_[mat_id + (row * cols_ + col) * mat_num_];
}

template <typename Scalar>
CUDAH MatrixDevice<Scalar> MatrixDeviceList<Scalar>::operator()(int mat_id)
{
	return MatrixDevice<Scalar>(rows_, cols_, mat_num_, buffer_ + mat_id);
}

template <typename Scalar>
void MatrixDeviceList<Scalar>::free()
{
	if (buffer_ != NULL && !is_copied_) {
		checkCudaErrors(cudaFree(buffer_));
	}

	mat_num_ = 0;
}

}

#endif


