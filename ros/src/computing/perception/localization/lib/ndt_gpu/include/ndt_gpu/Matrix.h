#ifndef GMAScalarRIX_H_
#define GMAScalarRIX_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include <float.h>

namespace gpu {

template <typename Scalar, int Rows, int Cols>
class Matrix {
public:
	CUDAH Matrix() {
		offset_ = 0;
		buffer_ = NULL;
		is_copied_ = false;
	}

	CUDAH Matrix(int offset, Scalar *buffer) {
		offset_ = offset;
		buffer_ = buffer;
		is_copied_ = true;
	}

	CUDAH Matrix(const Matrix<Scalar> &other) {
		offset_ = other.offset_;
		buffer_ = other.buffer_;
		is_copied_ = true;
	}

	CUDAH Matrix(Matrix<Scalar> &&other) {
		offset_ = other.offset_;
		buffer_ = other.buffer_;
		is_copied_ = false;

		other.offset_ = 0;
		other.buffer_ = NULL;
		other.is_copied_ = true;
	}

	CUDAH int rows() const { return Rows; }
	CUDAH int cols() const { return Cols; }
	CUDAH int offset() const { return offset_; }

	CUDAH Scalar *buffer() { return buffer_; }

	CUDAH void setOffset(int offset) { offset_ = offset; }
	CUDAH void setBuffer(Scalar *buffer) { buffer_ = buffer;}
	CUDAH void setCellVal(int row, int col, Scalar val) {
		buffer_[(row * Cols + col) * offset_] = val;
	}

	// Deep copy to output
	CUDAH void copy_from(const Matrix<Scalar, Rows, Cols> &output) {
#pragma unroll 1
		for (int i = 0; i < Rows; i++) {
#pragma unroll 1
			for (int j = 0; j < Cols; j++) {
				buffer_[(i * cols_ + j) * offset_] = output.at(i, j);
			}
		}
	}

	//Assignment operator
	// Copy assignment
	CUDAH Matrix<Scalar, Rows, Cols>& operator=(const Matrix<Scalar, Rows, Cols> &input) {
		offset_ = other.offset_;
		buffer_ = other.buffer_;
		is_copied_ = true;

		return *this;
	}

	CUDAH Scalar at(int row, int col) const { return buffer_[(row * Cols + col) * offset_]; }
	CUDAH Scalar at(int idx) const { return buffer_[idx * offset_]; }

	// Operators
	CUDAH Scalar& operator()(int row, int col) { return buffer_[(row * Cols + col) * offset_]; }
	CUDAH Scalar& operator()(int index) { return buffer_[idx * offset_]; }

	template <typename Scalar2>
	CUDAH Matrix<Scalar, Rows, Cols>& operator*=(Scalar2 val) {
#pragma unroll 1
		for (int i = 0; i < Rows; i++) {
#pragma unroll 1
			for (int j = 0; j < Cols; j++) {
				buffer_[(i * Cols + j) * offset_] *= val;
			}
		}

		return *this;
	}

	template <typename Scalar2>
	CUDAH Matrix<Scalar>& operator/=(Scalar2 val) {
#pragma unroll 1
		for (int i = 0; i < Rows; i++) {
#pragma unroll 1
			for (int j = 0; j < Cols; j++) {
				buffer_[(i * Cols + j) * offset_] /= val;
			}
		}

		return *this;
	}

	CUDAH void transpose(Matrix<Scalar, Rows, Cols> &output) {
		for (int i = 0; i < Rows; i++) {
			for (int j = 0; j < Cols; j++) {
				output(j, i) = buffer_[(i * Cols + j) * offset_];
			}
		}
	}

	//Only applicable for 3x3 matrix or below
	CUDAH bool inverse(Matrix<Scalar, Rows, Cols> &output);

	CUDAH Scalar dot(const Matrix<Scalar, Rows, Cols> &other) {
		Scalar res = 0;

#pragma unroll 1
		for (int i = 0; i < Rows; i++) {
#pragma unroll 1
			for (int j = 0; j < Cols; j++) {
				res += buffer_[(i * Rows + j) * offset_] * other.at(i, j);
			}
		}

		return res;
	}

	CUDAH Matrix<Scalar, Rows, 1> col(int index) {
		return Matrix<Scalar, Rows, 1>(offset_ * Cols, buffer_ + index * offset_);
	}

	CUDAH Matrix<Scalar, 1, Cols> row(int index) {
		return Matrix<Scalar, 1, Cols>(offset_, buffer_ + index * cols_ * offset_);
	}

	// Extract a col of RSize elements from (row, col)
	CUDAH Matrix<Scalar, RSize, 1> col(int row, int col) {
		return Matrix<Scalar, RSize, 1>(offset_ * Cols, buffer_ + (row * Cols + col) * offset_);
	}

	// Extract a row of CSize elements from (row, col)
	CUDAH Matrix<Scalar, 1, CSize> row(int row, int col) {
		return Matrix<Scalar, 1, CSize>(offset_, buffer_ + (row * Cols + col) * offset_);
	}

protected:
	Scalar *buffer_;
	int offset_;
	bool is_copied_;	// True: free buffer after being used, false: do nothing
};

template <>
CUDAH bool Matrix<double, 3, 3>::inverse(Matrix<double, 3, 3> &output)
{
	double det = at(0, 0) * at(1, 1) * at(2, 2) + at(0, 1) * at(1, 2) * at(2, 0) + at(1, 0) * at (2, 1) * at(0, 2)
					- at(0, 2) * at(1, 1) * at(2, 0) - at(0, 1) * at(1, 0) * at(2, 2) - at(0, 0) * at(1, 2) * at(2, 1);

	double idet = 1.0 / det;

	if (det != 0) {
		output(0, 0) = (at(1, 1) * at(2, 2) - at(1, 2) * at(2, 1)) * idet;
		output(0, 1) = - (at(0, 1) * at(2, 2) - at(0, 2) * at(2, 1)) * idet;
		output(0, 2) = (at(0, 1) * at(1, 2) - at(0, 2) * at(1, 1)) * idet;

		output(1, 0) = - (at(1, 0) * at(2, 2) - at(1, 2) * at(2, 0)) * idet;
		output(1, 1) = (at(0, 0) * at(2, 2) - at(0, 2) * at(2, 0)) * idet;
		output(1, 2) = - (at(0, 0) * at(1, 2) - at(0, 2) * at(1, 0)) * idet;

		output(2, 0) = (at(1, 0) * at(2, 1) - at(1, 1) * at(2, 0)) * idet;
		output(2, 1) = - (at(0, 0) * at(2, 1) - at(0, 1) * at(2, 0)) * idet;
		output(2, 2) = (at(0, 0) * at(1, 1) - at(0, 1) * at(1, 0)) * idet;

		return true;
	} else
		return false;
}


template class Matrix<float, 3, 1>;
template class Matrix<double, 3, 1>;
template class Matrix<double, 3, 3>;

}

#endif
