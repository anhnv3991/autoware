#include "ndt_gpu/Registration.h"
#include "ndt_gpu/debug.h"
#include <iostream>

namespace gpu {

template <typename PointSourceType, typename PointTargetType, typename Scalar>
GRegistration<PointSourceType, PointTargetType, Scalar>::GRegistration()
{
	max_iterations_ = 0;
	points_number_ = 0;

	converged_ = false;
	nr_iterations_ = 0;

	transformation_epsilon_ = 0;
	target_cloud_updated_ = true;
	target_points_number_ = 0;

	is_copied_ = false;

}

template <typename PointSourceType, typename PointTargetType, typename Scalar>
GRegistration<PointSourceType, PointTargetType, Scalar>::GRegistration(const GRegistration<PointSourceType, PointTargetType, Scalar> &other)
{
	transformation_epsilon_ = other.transformation_epsilon_;
	max_iterations_ = other.max_iterations_;

	//Original scanned point clouds
	source_cloud_ = other.source_cloud_;

	points_number_ = other.points_number_;

	trans_cloud_ = other.trans_cloud_;

	converged_ = other.converged_;

	nr_iterations_ = other.nr_iterations_;
	final_transformation_ = other.final_transformation_;
	transformation_ = other.transformation_;
	previous_transformation_ = other.previous_transformation_;

	target_cloud_updated_ = other.target_cloud_updated_;

	target_cloud_ = other.target_cloud_;

	target_points_number_ = other.target_points_number_;
	is_copied_ = true;
}

template <typename PointSourceType, typename PointTargetType, typename Scalar>
GRegistration<PointSourceType, PointTargetType, Scalar>::~GRegistration()
{
	source_cloud_.free();
	trans_cloud_.free();
	target_cloud_.free();
}

template <typename PointSourceType, typename PointTargetType, typename Scalar>
void GRegistration<PointSourceType, PointTargetType, Scalar>::setTransformationEpsilon(double trans_eps)
{
	transformation_epsilon_ = trans_eps;
}

template <typename PointSourceType, typename PointTargetType, typename Scalar>
double GRegistration<PointSourceType, PointTargetType, Scalar>::getTransformationEpsilon() const
{
	return transformation_epsilon_;
}

template <typename PointSourceType, typename PointTargetType, typename Scalar>
void GRegistration<PointSourceType, PointTargetType, Scalar>::setMaximumIterations(int max_itr)
{
	max_iterations_ = max_itr;
}

template <typename PointSourceType, typename PointTargetType, typename Scalar>
int GRegistration<PointSourceType, PointTargetType, Scalar>::getMaximumIterations() const
{
	return max_iterations_;
}

template <typename PointSourceType, typename PointTargetType, typename Scalar>
Eigen::Matrix<Scalar, 4, 4> GRegistration<PointSourceType, PointTargetType, Scalar>::getFinalTransformation() const
{
	return final_transformation_;
}

template <typename PointSourceType, typename PointTargetType, typename Scalar>
int GRegistration<PointSourceType, PointTargetType, Scalar>::getFinalNumIteration() const
{
	return nr_iterations_;
}

template <typename PointSourceType, typename PointTargetType, typename Scalar>
bool GRegistration<PointSourceType, PointTargetType, Scalar>::hasConverged() const
{
	return converged_;
}


template <typename PointSourceType, typename Scalar>
__global__ void convertInput(PointSourceType *input, MatrixDeviceList<Scalar> output, int point_num)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = idx; i < point_num; i += stride) {
		PointSourceType tmp = input[i];
		MatrixDevice<Scalar> out = output(i);
		out(0) = tmp.x;
		out(1) = tmp.y;
		out(2) = tmp.z;
	}
}

template <typename PointSourceType, typename PointTargetType, typename Scalar>
void GRegistration<PointSourceType, PointTargetType, Scalar>::setInputSource(typename pcl::PointCloud<PointSourceType>::Ptr input)
{
	//Convert point cloud to float x, y, z
	if (input->size() > 0) {
		points_number_ = input->size();

		PointSourceType *tmp;

		checkCudaErrors(cudaMalloc(&tmp, sizeof(PointSourceType) * points_number_));

		PointSourceType *host_tmp = input->points.data();

		// Pin the host buffer for accelerating the memory copy
#ifndef __aarch64__
		checkCudaErrors(cudaHostRegister(host_tmp, sizeof(PointSourceType) * points_number_, cudaHostRegisterDefault));
#endif

		checkCudaErrors(cudaMemcpy(tmp, host_tmp, sizeof(PointSourceType) * points_number_, cudaMemcpyHostToDevice));

		// Free old buffers
		source_cloud_.free();
		source_cloud_ = std::move(MatrixDeviceList<Scalar>(3, 1, points_number_));

		int block_x = (points_number_ > BLOCK_SIZE_X) ? BLOCK_SIZE_X : points_number_;
		int grid_x = (points_number_ - 1) / block_x + 1;

		convertInput<PointSourceType, Scalar><<<grid_x, block_x>>>(tmp, source_cloud_, points_number_);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		// Free old transformed points
		trans_cloud_.free();
		trans_cloud_ = std::move(MatrixDeviceList<Scalar>(3, 1, points_number_));

		// Initially, also copy scanned points to transformed buffers
		trans_cloud_.copy_from(source_cloud_);

		checkCudaErrors(cudaFree(tmp));

		// Unpin host buffer
#ifndef __aarch64__
		checkCudaErrors(cudaHostUnregister(host_tmp));
#endif
	}
}


//Set input MAP data
template <typename PointSourceType, typename PointTargetType, typename Scalar>
void GRegistration<PointSourceType, PointTargetType, Scalar>::setInputTarget(typename pcl::PointCloud<PointTargetType>::Ptr input)
{
	if (input->size() > 0) {
		target_points_number_ = input->size();

		PointTargetType *tmp;

		checkCudaErrors(cudaMalloc(&tmp, sizeof(PointTargetType) * target_points_number_));

		PointTargetType *host_tmp = input->points.data();

#ifndef __aarch64__
		checkCudaErrors(cudaHostRegister(host_tmp, sizeof(PointTargetType) * target_points_number_, cudaHostRegisterDefault));
#endif

		checkCudaErrors(cudaMemcpy(tmp, host_tmp, sizeof(PointTargetType) * target_points_number_, cudaMemcpyHostToDevice));

		// Free old target buffers
		target_cloud_.free();
		target_cloud_ = std::move(MatrixDeviceList<Scalar>(3, 1, target_points_number_));

		int block_x = (target_points_number_ > BLOCK_SIZE_X) ? BLOCK_SIZE_X : target_points_number_;
		int grid_x = (target_points_number_ - 1) / block_x + 1;

		// Convert from array of structures to three distinct arrays
		convertInput<PointTargetType, Scalar><<<grid_x, block_x>>>(tmp, target_cloud_, target_points_number_);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

#ifndef __aarch64__
		checkCudaErrors(cudaHostUnregister(host_tmp));
#endif
		checkCudaErrors(cudaFree(tmp));
	}
}

template <typename PointSourceType, typename PointTargetType, typename Scalar>
void GRegistration<PointSourceType, PointTargetType, Scalar>::align(const Eigen::Matrix<Scalar, 4, 4> &guess)
{
	converged_ = false;

	final_transformation_ = transformation_ = previous_transformation_ = Eigen::Matrix<Scalar, 4, 4>::Identity();

	computeTransformation(guess);
}

template <typename PointSourceType, typename PointTargetType, typename Scalar>
void GRegistration<PointSourceType, PointTargetType, Scalar>::computeTransformation(const Eigen::Matrix<Scalar, 4, 4> &guess) {
	printf("Unsupported by Registration\n");
}


template class GRegistration<pcl::PointXYZI, pcl::PointXYZI>;
template class GRegistration<pcl::PointXYZ, pcl::PointXYZ>;

}
