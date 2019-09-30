#ifndef MAScalarRIX_HOSScalar_H_
#define MAScalarRIX_HOSScalar_H_

#include "Matrix.h"
#include "MatrixDevice.h"

namespace gpu {
template <typename Scalar, int Rows, int Cols>
class MatrixHost : public Matrix<Scalar, Rows, Cols> {
public:
	MatrixHost();
	MatrixHost(int offset, Scalar *buffer);
	MatrixHost(const MatrixHost<Scalar, Rows, Cols> &other);
	MatrixHost(MatrixHost<Scalar, Rows, Cols> &&other);

	bool moveToGpu(MatrixDevice<Scalar, Rows, Cols> output);
	bool moveToHost(const MatrixDevice<Scalar, Rows, Cols> input);

	// Copy assignment
	MatrixHost<Scalar, Rows, Cols>& operator=(const MatrixHost<Scalar, Rows, Cols> &other);

	// Move assignment
	MatrixHost<Scalar, Rows, Cols>& operator=(MatrixHost<Scalar, Rows, Cols> &&other);

	void debug();

	template <typename Scalar2, int Rows2, int Cols2>
	friend std::ostream &operator<<(std::ostream &os, const MatrixHost<Scalar2, Rows2, Cols2> &value);

	~MatrixHost();

private:
	using Matrix<Scalar, Rows, Cols>::buffer_;
	using Matrix<Scalar, Rows, Cols>::offset_;
	using Matrix<Scalar, Rows, Cols>::is_copied_;
};



}

#endif
