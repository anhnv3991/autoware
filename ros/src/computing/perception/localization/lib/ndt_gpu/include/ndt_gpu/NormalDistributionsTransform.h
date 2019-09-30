#ifndef GPU_NDT_H_
#define GPU_NDT_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "Registration.h"
#include "common.h"
#include "VoxelGrid.h"
#include "Eigen/Geometry"

namespace gpu {
template <typename PointSourceType, typename PointTargetType>
class GNormalDistributionsTransform: public GRegistration<PointSourceType, PointTargetType> {
public:
	GNormalDistributionsTransform();

	GNormalDistributionsTransform(const GNormalDistributionsTransform<PointSourceType, PointTargetType> &other);

	void setStepSize(double step_size);

	void setResolution(float resolution);

	void setOulierRatio(double olr);

	double getStepSize() const;

	float getResolution() const;

	double getOulierRatio() const;

	double getTransformationProbability() const;

	int getRealIterations();

	/* Set the input map points */
	void setInputTarget(typename pcl::PointCloud<PointTargetType>::Ptr input);

	/* Compute and get fitness score */
	double getFitnessScore(double max_range = DBL_MAX);

	~GNormalDistributionsTransform();

protected:
	void computeTransformation(const Eigen::Matrix<float, 4, 4> &guess);
	double computeDerivatives(Eigen::Matrix<double, 6, 1> &score_gradient, Eigen::Matrix<double, 6, 6> &hessian,
								MatrixDeviceList<float, 3, 1> trans_cloud,
								int points_num, Eigen::Matrix<double, 6, 1> pose, bool compute_hessian = true);

	using GRegistration<PointSourceType, PointTargetType>::transformation_epsilon_;
	using GRegistration<PointSourceType, PointTargetType>::max_iterations_;
	using GRegistration<PointSourceType, PointTargetType>::source_cloud_;
	using GRegistration<PointSourceType, PointTargetType>::points_number_;
	using GRegistration<PointSourceType, PointTargetType>::trans_cloud_;
	using GRegistration<PointSourceType, PointTargetType>::converged_;
	using GRegistration<PointSourceType, PointTargetType>::nr_iterations_;
	using GRegistration<PointSourceType, PointTargetType>::final_transformation_;
	using GRegistration<PointSourceType, PointTargetType>::transformation_;
	using GRegistration<PointSourceType, PointTargetType>::previous_transformation_;
	using GRegistration<PointSourceType, PointTargetType>::target_cloud_updated_;
	using GRegistration<PointSourceType, PointTargetType>::target_cloud_;
	using GRegistration<PointSourceType, PointTargetType>::target_points_number_;
	using GRegistration<PointSourceType, PointTargetType>::is_copied_;
private:
	//Copied from ndt.h
    double auxilaryFunction_PsiMT (double a, double f_a, double f_0, double g_0, double mu = 1.e-4);

    //Copied from ndt.h
    double auxilaryFunction_dPsiMT (double g_a, double g_0, double mu = 1.e-4);

    double updateIntervalMT (double &a_l, double &f_l, double &g_l,
								double &a_u, double &f_u, double &g_u,
								double a_t, double f_t, double g_t);

    double trialValueSelectionMT (double a_l, double f_l, double g_l,
									double a_u, double f_u, double g_u,
									double a_t, double f_t, double g_t);

	void transformPointCloud(MatrixDeviceList<float, 3, 1> input, MatrixDeviceList<float, 3, 1> output,
								int points_number, Eigen::Matrix<float, 4, 4> transform);

	void computeAngleDerivatives(MatrixHost<double, 6, 1> pose, bool compute_hessian = true);

	double computeStepLengthMT(const Eigen::Matrix<double, 6, 1> &x, Eigen::Matrix<double, 6, 1> &step_dir,
								double step_init, double step_max, double step_min, double &score,
								Eigen::Matrix<double, 6, 1> &score_gradient, Eigen::Matrix<double, 6, 6> &hessian,
								MatrixDeviceList<float, 3, 1> trans_cloud, int points_num);

	void computeHessian(Eigen::Matrix<double, 6, 6> &hessian, MatrixDeviceList<float, 3, 1> trans_cloud, int points_num, Eigen::Matrix<double, 6, 1> &p);


	double gauss_d1_, gauss_d2_;
	double outlier_ratio_;
	//MatrixHost j_ang_a_, j_ang_b_, j_ang_c_, j_ang_d_, j_ang_e_, j_ang_f_, j_ang_g_, j_ang_h_;
	MatrixHost<double> j_ang_;

	//MatrixHost h_ang_a2_, h_ang_a3_, h_ang_b2_, h_ang_b3_, h_ang_c2_, h_ang_c3_, h_ang_d1_, h_ang_d2_, h_ang_d3_,
	//			h_ang_e1_, h_ang_e2_, h_ang_e3_, h_ang_f1_, h_ang_f2_, h_ang_f3_;
	MatrixHost<double> h_ang_;


	//MatrixDevice dj_ang_a_, dj_ang_b_, dj_ang_c_, dj_ang_d_, dj_ang_e_, dj_ang_f_, dj_ang_g_, dj_ang_h_;
	MatrixDevice<double> dj_ang_;


	//MatrixDevice dh_ang_a2_, dh_ang_a3_, dh_ang_b2_, dh_ang_b3_, dh_ang_c2_, dh_ang_c3_, dh_ang_d1_, dh_ang_d2_, dh_ang_d3_,
	//			dh_ang_e1_, dh_ang_e2_, dh_ang_e3_, dh_ang_f1_, dh_ang_f2_, dh_ang_f3_;
	MatrixDevice<double> dh_ang_;

	double step_size_;
	float resolution_;
	double trans_probability_;

	int real_iterations_;


	GVoxelGrid voxel_grid_;
};

}

#endif
