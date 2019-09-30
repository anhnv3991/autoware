#ifndef MATRIX_DEVICE_H_
#define MATRIX_DEVICE_H_

#include "Matrix.h"
#include <iostream>

namespace gpu {
template <typename Scalar, int Rows, int Cols>
class MatrixDevice : public Matrix<Scalar, Rows, Cols> {
public:
	MatrixDevice();

	CUDAH MatrixDevice(int offset, Scalar *buffer) :  Matrix<Scalar, Rows, Cols>(offset, buffer){}

	CUDAH MatrixDevice(const MatrixDevice<Scalar, Rows, Cols> &other) {
		buffer_ = other.buffer_;
		offset_ = other.offset_;
		is_copied_ = true;
	}

	CUDAH bool isEmpty() {
		return (Rows == 0 || Cols == 0 || buffer_ == NULL);
	}

	CUDAH MatrixDevice<Scalar, Rows, 1> col(int index) {
		return MatrixDevice<Scalar, Rows, 1>(offset_ * Cols, buffer_ + index * offset_);
	}

	CUDAH MatrixDevice<Scalar, 1, Cols> row(int index) {
		return MatrixDevice<Scalar, 1, Cols>(offset_, buffer_ + index * Cols * offset_);
	}

	CUDAH MatrixDevice<Scalar, RSize, 1> col(int row, int col, int rsize) {
		return MatrixDevice<Scalar, RSize, 1>(offset_ * Cols, buffer_ + (row * Cols + col) * offset_);
	}

	// Extract a row of CSize elements from (row, col)
	CUDAH MatrixDevice<Scalar, 1, CSize> row(int row, int col) {
		return MatrixDevice<Scalar, 1, CSize>(offset_, buffer_ + (row * Cols + col) * offset_);
	}

	CUDAH MatrixDevice<Scalar, Rows, Cols>& operator=(const MatrixDevice<Scalar, Rows, Cols> &other) {
		buffer_ = other.buffer_;
		offset_ = other.offset_;
		is_copied_ = true;

		return *this;
	}

	CUDAH MatrixDevice<Scalar, Rows, Cols>& operator=(MatrixDevice<Scalar, Rows, Cols> &&other) {
		buffer_ = other.buffer_;
		offset_ = other.offset_;
		is_copied_ = false;

		other.buffer_ = nullptr;
		other.offset_ = 0;
		other.is_copied_ = true;

		return *this;
	}

	CUDAH void copy_from(const MatrixDevice<Scalar, Rows, Cols> &other) {
#pragma unroll 1
		for (int i = 0; i < Rows; i++) {
#pragma unroll 1
			for (int j = 0; j < Cols; j++) {
				buffer_[(i * Cols + j) * offset_] = other.at(i, j);
			}
		}
	}

	CUDAH void setBuffer(Scalar *buffer) { buffer_ = buffer; }

	void free();
private:
	using Matrix<Scalar, Rows, Cols>::buffer_;
	using Matrix<Scalar, Rows, Cols>::offset_;
	using Matrix<Scalar, Rows, Cols>::is_copied_;
};

}

#endif
