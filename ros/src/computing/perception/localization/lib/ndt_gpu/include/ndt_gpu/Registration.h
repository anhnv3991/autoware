#ifndef GNDT_H_
#define GNDT_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "Matrix.h"
#include "MatrixHost.h"
#include "MatrixDevice.h"
#include "MatrixDeviceList.h"
#include "common.h"
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

namespace gpu {
template <typename PointSourceType, typename PointTargetType, typename Scalar = float>
class GRegistration {
public:
	GRegistration();
	GRegistration(const GRegistration<PointSourceType, PointTargetType, Scalar> &other);

	void align(const Eigen::Matrix<Scalar, 4, 4> &guess);

	void setTransformationEpsilon(double trans_eps);

	double getTransformationEpsilon() const;

	void setMaximumIterations(int max_itr);

	int getMaximumIterations() const;

	Eigen::Matrix<Scalar, 4, 4> getFinalTransformation() const;

	/* Set input Scanned point cloud.
	 * Copy input points from the main memory to the GPU memory */
	void setInputSource(typename pcl::PointCloud<PointSourceType>::Ptr input);

	/* Set input reference map point cloud.
	 * Copy input points from the main memory to the GPU memory */
	void setInputTarget(typename pcl::PointCloud<PointTargetType>::Ptr input);

	int getFinalNumIteration() const;

	bool hasConverged() const;

	virtual ~GRegistration();
protected:

	virtual void computeTransformation(const Eigen::Matrix<Scalar, 4, 4> &guess);

	double transformation_epsilon_;
	int max_iterations_;

	//Source scanned point clouds
	MatrixDeviceList<Scalar> source_cloud_;
	int points_number_;

	//Transformed source point clouds
	MatrixDeviceList<Scalar> trans_cloud_;

	bool converged_;
	int nr_iterations_;

	Eigen::Matrix<Scalar, 4, 4> final_transformation_, transformation_, previous_transformation_;

	bool target_cloud_updated_;

	// Target cloud
	MatrixDeviceList<Scalar> target_cloud_;
	int target_points_number_;

	bool is_copied_;
};

}

#endif
