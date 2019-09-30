#include "ndt_gpu/NormalDistributionsTransform.h"
#include "ndt_gpu/debug.h"
#include "ndt_gpu/MatrixDevice.h"
#include "ndt_gpu/MatrixDeviceList.h"
#include "ndt_gpu/VectorRegister.h"
#include <cmath>
#include <iostream>
#include <pcl/common/transforms.h>

namespace gpu {

template <typename PointSourceType, typename PointTargetType>
GNormalDistributionsTransform<PointSourceType, PointTargetType>::GNormalDistributionsTransform()
{
	gauss_d1_ = gauss_d2_ = 0;
	outlier_ratio_ = 0.55;
	step_size_ = 0.1;
	resolution_ = 1.0f;
	trans_probability_ = 0;

	double gauss_c1, gauss_c2, gauss_d3;

	// Initializes the guassian fitting parameters (eq. 6.8) [Magnusson 2009]
	gauss_c1 = 10.0 * (1 - outlier_ratio_);
	gauss_c2 = outlier_ratio_ / pow (resolution_, 3);
	gauss_d3 = -log (gauss_c2);
	gauss_d1_ = -log ( gauss_c1 + gauss_c2 ) - gauss_d3;
	gauss_d2_ = -2 * log ((-log ( gauss_c1 * exp ( -0.5 ) + gauss_c2 ) - gauss_d3) / gauss_d1_);

	transformation_epsilon_ = 0.1;
	max_iterations_ = 35;

	j_ang_ = MatrixHost<double>(24, 1);

	h_ang_ = MatrixHost<double>(45, 1);

	dj_ang_ = MatrixDevice<double>(24, 1);

	dh_ang_ = MatrixDevice<double>(45, 1);

	real_iterations_ = 0;
}

template <typename PointSourceType, typename PointTargetType>
GNormalDistributionsTransform<PointSourceType, PointTargetType>::GNormalDistributionsTransform(const GNormalDistributionsTransform<PointSourceType, PointTargetType> &other) :
GRegistration<PointSourceType, PointTargetType>(other)
{
	gauss_d1_ = other.gauss_d1_;
	gauss_d2_ = other.gauss_d2_;

	outlier_ratio_ = other.outlier_ratio_;

	j_ang_ = other.j_ang_;
	h_ang_ = other.h_ang_;
	dj_ang_ = other.dj_ang_;
	dh_ang_ = other.dh_ang_;

	step_size_ = other.step_size_;
	resolution_ = other.resolution_;
	trans_probability_ = other.trans_probability_;
	real_iterations_ = other.real_iterations_;

	voxel_grid_ = other.voxel_grid_;
}

template <typename PointSourceType, typename PointTargetType>
GNormalDistributionsTransform<PointSourceType, PointTargetType>::~GNormalDistributionsTransform()
{
	dj_ang_.free();
	dh_ang_.free();

}

template <typename PointSourceType, typename PointTargetType>
void GNormalDistributionsTransform<PointSourceType, PointTargetType>::setStepSize(double step_size)
{
	step_size_ = step_size;
}

template <typename PointSourceType, typename PointTargetType>
void GNormalDistributionsTransform<PointSourceType, PointTargetType>::setResolution(float resolution)
{
	resolution_ = resolution;
}

template <typename PointSourceType, typename PointTargetType>
void GNormalDistributionsTransform<PointSourceType, PointTargetType>::setOulierRatio(double olr)
{
	outlier_ratio_ = olr;
}

template <typename PointSourceType, typename PointTargetType>
double GNormalDistributionsTransform<PointSourceType, PointTargetType>::getStepSize() const
{
	return step_size_;
}

template <typename PointSourceType, typename PointTargetType>
float GNormalDistributionsTransform<PointSourceType, PointTargetType>::getResolution() const
{
	return resolution_;
}

template <typename PointSourceType, typename PointTargetType>
double GNormalDistributionsTransform<PointSourceType, PointTargetType>::getOulierRatio() const
{
	return outlier_ratio_;
}

template <typename PointSourceType, typename PointTargetType>
double GNormalDistributionsTransform<PointSourceType, PointTargetType>::getTransformationProbability() const
{
	return trans_probability_;
}

template <typename PointSourceType, typename PointTargetType>
int GNormalDistributionsTransform<PointSourceType, PointTargetType>::getRealIterations()
{
	 return real_iterations_;
}

template <typename PointSourceType, typename PointTargetType>
double GNormalDistributionsTransform<PointSourceType, PointTargetType>::auxilaryFunction_PsiMT(double a, double f_a, double f_0, double g_0, double mu)
{
  return (f_a - f_0 - mu * g_0 * a);
}

template <typename PointSourceType, typename PointTargetType>
double GNormalDistributionsTransform<PointSourceType, PointTargetType>::auxilaryFunction_dPsiMT(double g_a, double g_0, double mu)
{
  return (g_a - mu * g_0);
}

template <typename PointSourceType, typename PointTargetType>
void GNormalDistributionsTransform<PointSourceType, PointTargetType>::setInputTarget(typename pcl::PointCloud<PointTargetType>::Ptr input)
{
	// Copy input map data from the host memory to the GPU memory
	GRegistration<PointSourceType, PointTargetType>::setInputTarget(input);

	// Build the voxel grid
	if (target_points_number_ != 0) {
		voxel_grid_.setLeafSize(resolution_, resolution_, resolution_);
		voxel_grid_.setInput(target_cloud_, target_points_number_);
	}
}

template <typename PointSourceType, typename PointTargetType>
void GNormalDistributionsTransform<PointSourceType, PointTargetType>::computeTransformation(const Eigen::Matrix<float, 4, 4> &guess)
{

	nr_iterations_ = 0;
	converged_ = false;

	double gauss_c1, gauss_c2, gauss_d3;

	gauss_c1 = 10 * ( 1 - outlier_ratio_);
	gauss_c2 = outlier_ratio_ / pow(resolution_, 3);
	gauss_d3 = - log(gauss_c2);
	gauss_d1_ = -log(gauss_c1 + gauss_c2) - gauss_d3;
	gauss_d2_ = -2 * log((-log(gauss_c1 * exp(-0.5) + gauss_c2) - gauss_d3) / gauss_d1_);

	if (guess != Eigen::Matrix4f::Identity()) {
		final_transformation_ = guess;

		transformPointCloud(source_cloud_, trans_cloud_, points_number_, guess);
	}

	Eigen::Transform<float, 3, Eigen::Affine, Eigen::ColMajor> eig_transformation;
	eig_transformation.matrix() = final_transformation_;

	Eigen::Matrix<double, 6, 1> p, delta_p, score_gradient;
	Eigen::Vector3f init_translation = eig_transformation.translation();
	Eigen::Vector3f init_rotation = eig_transformation.rotation().eulerAngles(0, 1, 2);

	p << init_translation(0), init_translation(1), init_translation(2), init_rotation(0), init_rotation(1), init_rotation(2);

	Eigen::Matrix<double, 6, 6> hessian;

	double score = 0;
	double delta_p_norm;

	score = computeDerivatives(score_gradient, hessian, trans_cloud_, points_number_, p);

	int loop_time = 0;

	while (!converged_) {
		previous_transformation_ = transformation_;

		Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6>> sv(hessian, Eigen::ComputeFullU | Eigen::ComputeFullV);

		delta_p = sv.solve(-score_gradient);

		delta_p_norm = delta_p.norm();

		if (delta_p_norm == 0 || delta_p_norm != delta_p_norm) {

			trans_probability_ = score / static_cast<double>(points_number_);
			converged_ = delta_p_norm == delta_p_norm;
			return;
		}

		delta_p.normalize();
		delta_p_norm = computeStepLengthMT(p, delta_p, delta_p_norm, step_size_, transformation_epsilon_ / 2, score, score_gradient, hessian, trans_cloud_, points_number_);

		delta_p *= delta_p_norm;

		transformation_ = (Eigen::Translation<float, 3>(static_cast<float>(delta_p(0)), static_cast<float>(delta_p(1)), static_cast<float>(delta_p(2))) *
							Eigen::AngleAxis<float>(static_cast<float>(delta_p(3)), Eigen::Vector3f::UnitX()) *
							Eigen::AngleAxis<float>(static_cast<float>(delta_p(4)), Eigen::Vector3f::UnitY()) *
							Eigen::AngleAxis<float>(static_cast<float>(delta_p(5)), Eigen::Vector3f::UnitZ())).matrix();

		p = p + delta_p;

		//Not update visualizer

		if (nr_iterations_ > max_iterations_ || (nr_iterations_ && (std::fabs(delta_p_norm) < transformation_epsilon_)))
			converged_ = true;

		nr_iterations_++;

		loop_time++;
	}

	trans_probability_ = score / static_cast<double>(points_number_);
}

/* Added on 2019/09/20 */
__global__ void computePointGradient(MatrixDeviceList<float> points, int points_num,
										int *valid_points, int valid_points_num,
										double *dj_ang,
										MatrixDeviceList<double> point_gradients,	// 3x6 matrix
										int2 *mat_list, int mat_size)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	__shared__ double j_ang[24];

	if (threadIdx.x < 24) {
		j_ang[threadIdx.x] = dj_ang[threadIdx.x];
	}
	__syncthreads();

	for (int i = id; i < valid_points_num; i += stride) {
		int pid = valid_points[i];
		MatrixDevice<double> pg = point_gradients(pid);
		VectorR<double> p = points(pid);

		//Set the 3x3 block start from (0, 0) to identity matrix
		pg(0, 0) = 1;
		pg(1, 1) = 1;
		pg(2, 2) = 1;

		for (int k = 0; k < mat_size; k++) {
			int2 ids = mat_list[k];
			pg(ids.x, ids.y) = p(0) * j_ang[k * 3] + p(1) * j_ang[k * 3 + 1] + p(2) * j_ang[k * 3 + 2];
		}
	}
}
/* End of adding */



/* Added 2019/09/21 */
__global__ void computePointHessian(MatrixDeviceList<float> points, int points_num,
										int *valid_points, int valid_points_num,
										double *dh_ang,
										MatrixDeviceList<double> point_hessians, // 18x6 matrix
										int2 *mat_list, int mat_size)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	__shared__ double h_ang[45];

	if (threadIdx.x < 45) {
		h_ang[threadIdx.x] = dh_ang[threadIdx.x];
	}

	__syncthreads();

	for (int i = id; i < valid_points_num; i += stride) {
		int pid = valid_points[i];
		MatrixDevice<double> ph = point_hessians(pid);
		VectorR<double> p = points(pid);

		for (int k = 0; k < mat_size; k++) {
			int2 ids = mat_list[k];
			ph(ids.x, ids.y) = p(0) * h_ang[k * 3] + p(1) * h_ang[k * 3 + 1] + p(2) * h_ang[k * 3 + 2];
		}
	}
}
/* End of adding */


/* compute score_inc list for input points.
 * The final score_inc is calculated by a reduction sum
 * on this score_inc list. */
__global__ void computeScoreList(int *starting_voxel_id, int *voxel_id, int valid_points_num,
												double *e_x_cov_x, double gauss_d1, double *score)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = id; i < valid_points_num; i += stride) {

		double score_inc = 0;

		for (int vid = starting_voxel_id[i]; vid < starting_voxel_id[i + 1]; vid++) {
			double tmp_ex = e_x_cov_x[vid];

			score_inc += (tmp_ex > 1 || tmp_ex < 0 || tmp_ex != tmp_ex) ? 0 : -gauss_d1 * tmp_ex;
		}

		score[i] = score_inc;
	}
}

/* Adding on 2019/09/24 */
__global__ void computeScoreGradientListV2(MatrixDeviceList<float> trans_points,
											int *valid_points,
											int *starting_voxel_id, int *voxel_id, int valid_points_num,
											MatrixDeviceList<double> centroids,
											int voxel_num, double *e_x_cov_x,
											MatrixDeviceList<double> cov_dxd_pi,	// 3x6 matrix
											double gauss_d1, int valid_voxel_num,
											MatrixDeviceList<double> score_gradients)	// 6x1 matrix
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	int col = blockIdx.y;

	if (col < 6) {
		for (int i = id; i < valid_points_num; i += stride) {
			int pid = valid_points[i];
			VectorR<double> p = trans_points(pid);
			MatrixDevice<double> sg = score_gradients(i);

			double tmp_sg = 0.0;

			for ( int j = starting_voxel_id[i]; j < starting_voxel_id[i + 1]; j++) {
				int vid = voxel_id[j];
				VectorR<double> centr = centroids(vid);
				MatrixDevice<double> cov_dxd_pi_mat = cov_dxd_pi(j).col(col);
				double tmp_ex = e_x_cov_x[j];

				if (!(tmp_ex > 1 || tmp_ex < 0 || tmp_ex != tmp_ex)) {
					tmp_ex *= gauss_d1;

					tmp_sg += ((p(0) - centr(0)) * cov_dxd_pi_mat(0) + (p(1) - centr(1)) * cov_dxd_pi_mat(1) + (p(2) - centr(2)) * cov_dxd_pi_mat(2)) * tmp_ex;
				}
			}

			sg(col) = tmp_sg;
		}
	}
}
/* End of adding */

/* First step to compute score gradient list for input points */
__global__ void computeScoreGradientList(MatrixDeviceList<float> trans_points,
											int *valid_points,
											int *starting_voxel_id, int *voxel_id, int valid_points_num,
											MatrixDeviceList<double> centroids,
											int voxel_num, double *e_x_cov_x,
											double *cov_dxd_pi, double gauss_d1, int valid_voxel_num,
											double *score_gradients)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	int col = blockIdx.y;

	if (col < 6) {
		double *sg = score_gradients + col * valid_points_num;
		double *cov_dxd_pi_mat0 = cov_dxd_pi + col * valid_voxel_num;
		double *cov_dxd_pi_mat1 = cov_dxd_pi_mat0 + 6 * valid_voxel_num;
		double *cov_dxd_pi_mat2 = cov_dxd_pi_mat1 + 6 * valid_voxel_num;

		for (int i = id; i < valid_points_num; i += stride) {
			int pid = valid_points[i];
			MatrixDevice<float> p = trans_points(pid);
			double d_x = static_cast<double>(p(0));
			double d_y = static_cast<double>(p(1));
			double d_z = static_cast<double>(p(2));

			double tmp_sg = 0.0;

			for ( int j = starting_voxel_id[i]; j < starting_voxel_id[i + 1]; j++) {
				int vid = voxel_id[j];
				MatrixDevice<double> centr = centroids(vid);
				double tmp_ex = e_x_cov_x[j];

				if (!(tmp_ex > 1 || tmp_ex < 0 || tmp_ex != tmp_ex)) {
					tmp_ex *= gauss_d1;

					tmp_sg += ((d_x - centr(0)) * cov_dxd_pi_mat0[j] + (d_y - centr(1)) * cov_dxd_pi_mat1[j] + (d_z - centr(2)) * cov_dxd_pi_mat2[j]) * tmp_ex;
				}
			}

			sg[i] = tmp_sg;
		}
	}
}

/* Intermediate step to compute e_x_cov_x */
/* Adding on 2019/09/24 */
__global__ void computeExCovXV2(MatrixDeviceList<float> trans_cloud, int *valid_points,
								int *starting_voxel_id, int *voxel_id, int valid_points_num,
								MatrixDeviceList<double> centroid,
								double gauss_d1, double gauss_d2,
								double *e_x_cov_x,
								MatrixDeviceList<double> inverse_covariance)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = id; i < valid_points_num; i += stride) {
		int pid = valid_points[i];
		MatrixDevice<float> p = trans_cloud(pid);
		double d_x = static_cast<double>(p(0));
		double d_y = static_cast<double>(p(1));
		double d_z = static_cast<double>(p(2));
		double t_x, t_y, t_z;


		for ( int j = starting_voxel_id[i]; j < starting_voxel_id[i + 1]; j++) {
			int vid = voxel_id[j];
			MatrixDevice<double> centr = centroid(vid);
			MatrixDevice<double> icov = inverse_covariance(vid);

			t_x = d_x - centr(0);
			t_y = d_y - centr(1);
			t_z = d_z - centr(2);

			e_x_cov_x[j] =  exp(-gauss_d2 * ((t_x * icov(0, 0) + t_y * icov(0, 1) + t_z * icov(0, 2)) * t_x
										+ ((t_x * icov(1, 0) + t_y * icov(1, 1) + t_z * icov(1, 2)) * t_y)
										+ ((t_x * icov(2, 0) + t_y * icov(2, 1) + t_z * icov(2, 2)) * t_z)) / 2.0);
		}
	}
}

/* End of adding */

__global__ void computeExCovX(MatrixDeviceList<float> trans_cloud, int *valid_points,
								int *starting_voxel_id, int *voxel_id, int valid_points_num,
								MatrixDeviceList<double> centroid,
								double gauss_d1, double gauss_d2,
								double *e_x_cov_x,
								double *icov00, double *icov01, double *icov02,
								double *icov10, double *icov11, double *icov12,
								double *icov20, double *icov21, double *icov22)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = id; i < valid_points_num; i += stride) {
		int pid = valid_points[i];
		MatrixDevice<float, 3, 1> p = trans_cloud(pid);
		double d_x = static_cast<double>(p(0));
		double d_y = static_cast<double>(p(1));
		double d_z = static_cast<double>(p(2));
		double t_x, t_y, t_z;


		for ( int j = starting_voxel_id[i]; j < starting_voxel_id[i + 1]; j++) {
			int vid = voxel_id[j];
			MatrixDevice<double, 3, 1> centr = centroid(vid);

			t_x = d_x - centr(0);
			t_y = d_y - centr(1);
			t_z = d_z - centr(2);

			e_x_cov_x[j] =  exp(-gauss_d2 * ((t_x * icov00[vid] + t_y * icov01[vid] + t_z * icov02[vid]) * t_x
										+ ((t_x * icov10[vid] + t_y * icov11[vid] + t_z * icov12[vid]) * t_y)
										+ ((t_x * icov20[vid] + t_y * icov21[vid] + t_z * icov22[vid]) * t_z)) / 2.0);
		}
	}
}

/* update e_x_cov_x - Reusable portion of Equation 6.12 and 6.13 [Magnusson 2009] */
__global__ void updateExCovX(double *e_x_cov_x, double gauss_d2, int valid_voxel_num)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = id; i < valid_voxel_num; i += stride) {
		e_x_cov_x[i] *= gauss_d2;
	}
}

/* compute cov_dxd_pi as reusable portion of Equation 6.12 and 6.13 [Magnusson 2009]*/
__global__ void computeCovDxdPi(int *valid_points, int *starting_voxel_id, int *voxel_id, int valid_points_num,
											double *inverse_covariance, int voxel_num,
											double gauss_d1, double gauss_d2, double *point_gradients,
											double *cov_dxd_pi, int valid_voxel_num)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	int row = blockIdx.y;
	int col = blockIdx.z;

	if (row < 3 && col < 6) {
		double *icov0 = inverse_covariance + row * 3 * voxel_num;
		double *icov1 = icov0 + voxel_num;
		double *icov2 = icov1 + voxel_num;
		double *cov_dxd_pi_tmp = cov_dxd_pi + (row * 6 + col) * valid_voxel_num;
		double *pg_tmp0 = point_gradients + col * valid_points_num;
		double *pg_tmp1 = pg_tmp0 + 6 * valid_points_num;
		double *pg_tmp2 = pg_tmp1 + 6 * valid_points_num;

		for (int i = id; i < valid_points_num; i += stride) {
			double pg0 = pg_tmp0[i];
			double pg1 = pg_tmp1[i];
			double pg2 = pg_tmp2[i];

			for ( int j = starting_voxel_id[i]; j < starting_voxel_id[i + 1]; j++) {
				int vid = voxel_id[j];

				cov_dxd_pi_tmp[j] = icov0[vid] * pg0 + icov1[vid] * pg1 + icov2[vid] * pg2;
			}
	}
	}
}

/* Added 2019/09/23 */
__global__ void computeHessianListS0(MatrixDeviceList<float> trans_cloud,
										int *valid_points,
										int *starting_voxel_id, int *voxel_id, int valid_points_num,
										MatrixDeviceList<double> centroid,
										MatrixDeviceList<double> inverse_covariance,	//3x3
										MatrixDeviceList<double> point_gradients,		//3x6
										MatrixDeviceList<double> tmp_hessian,			//6x1
										int valid_voxel_num)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	int col = blockIdx.y;

	if (col < 6) {
		for (int i = id; i < valid_points_num; i += stride) {
			int pid = valid_points[i];
			MatrixDevice<float> p = trans_cloud(pid);
			double d_x = static_cast<double>(p(0));
			double d_y = static_cast<double>(p(1));
			double d_z = static_cast<double>(p(2));
			MatrixDevice<double> pg = point_gradients(i);

			double pg0 = pg(0, col);
			double pg1 = pg(1, col);
			double pg2 = pg(2, col);

			for ( int j = starting_voxel_id[i]; j < starting_voxel_id[i + 1]; j++) {
				int vid = voxel_id[j];
				MatrixDevice<double> tmp_h = tmp_hessian(j);
				MatrixDevice<double> centr = centroid(vid);
				MatrixDevice<double> icov = inverse_covariance(vid);
				double t_hx = 0.0, t_hy = 0.0, t_hz = 0.0;

				for (int k = 0; k < 3; k++) {
					t_hx += icov(0, i) * pg(i, col);
				}

				t_hx *= (d_x - centr(0));

				for (int k = 0; k < 3; k++) {
					t_hy += icov(1, i) * pg(i, col);
				}

				t_hy *= (d_y - centr(1));

				for (int k = 0; k < 3; k++) {
					t_hz += icov(2, i) * pg(i, col);
				}

				t_hz *= (d_z - centr(2));

				tmp_h(col) = t_hx + t_hy + t_hz;			}
		}
	}
}
/* End of adding */


/* First step to compute hessian list for input points */
__global__ void computeHessianListS0(float *trans_x, float *trans_y, float *trans_z,
													int *valid_points,
													int *starting_voxel_id, int *voxel_id, int valid_points_num,
													double *centroid_x, double *centroid_y, double *centroid_z,
													double *icov00, double *icov01, double *icov02,
													double *icov10, double *icov11, double *icov12,
													double *icov20, double *icov21, double *icov22,
													double *point_gradients,
													double *tmp_hessian,
													int valid_voxel_num)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	int col = blockIdx.y;

	if (col < 6) {
		double *tmp_pg0 = point_gradients + col * valid_points_num;
		double *tmp_pg1 = tmp_pg0 + 6 * valid_points_num;
		double *tmp_pg2 = tmp_pg1 + 6 * valid_points_num;
		double *tmp_h = tmp_hessian + col * valid_voxel_num;

		for (int i = id; i < valid_points_num; i += stride) {
			int pid = valid_points[i];
			double d_x = static_cast<double>(trans_x[pid]);
			double d_y = static_cast<double>(trans_y[pid]);
			double d_z = static_cast<double>(trans_z[pid]);

			double pg0 = tmp_pg0[i];
			double pg1 = tmp_pg1[i];
			double pg2 = tmp_pg2[i];

			for ( int j = starting_voxel_id[i]; j < starting_voxel_id[i + 1]; j++) {
				int vid = voxel_id[j];

				tmp_h[j] = (d_x - centroid_x[vid]) * (icov00[vid] * pg0 + icov01[vid] * pg1 + icov02[vid] * pg2)
							+ (d_y - centroid_y[vid]) * (icov10[vid] * pg0 + icov11[vid] * pg1 + icov12[vid] * pg2)
							+ (d_z - centroid_z[vid]) * (icov20[vid] * pg0 + icov21[vid] * pg1 + icov22[vid] * pg2);
			}
	}
	}
}

/* Added on 2019/09/24 */
__global__ void computeHessianListS1(MatrixDeviceList<float> trans_cloud,
										int *valid_points,
										int *starting_voxel_id, int *voxel_id, int valid_points_num,
										MatrixDeviceList<double> centroids,
										double gauss_d1, double gauss_d2, MatrixDeviceList<double> hessians,	//6x6
										double *e_x_cov_x, MatrixDeviceList<double> tmp_hessian,	//6x1
										double *cov_dxd_pi,
										MatrixDeviceList<double> point_gradients,	//3x6
										int valid_voxel_num)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	int row = blockIdx.y;
	int col = blockIdx.z;

	if (row < 6 && col < 6) {
		double *cov_dxd_pi_mat0 = cov_dxd_pi + row * valid_voxel_num;
		double *cov_dxd_pi_mat1 = cov_dxd_pi_mat0 + 6 * valid_voxel_num;
		double *cov_dxd_pi_mat2 = cov_dxd_pi_mat1 + 6 * valid_voxel_num;

		for (int i = id; i < valid_points_num; i += stride) {
			int pid = valid_points[i];
			MatrixDevice<float> p = trans_cloud(pid);
			double d_x = static_cast<double>(p(0));
			double d_y = static_cast<double>(p(1));
			double d_z = static_cast<double>(p(2));
			MatrixDevice<double> tmp_pg = point_gradients(i).col(col);

			double pg0 = tmp_pg(0);
			double pg1 = tmp_pg(1);
			double pg2 = tmp_pg(2);

			double final_hessian = 0.0;

			for ( int j = starting_voxel_id[i]; j < starting_voxel_id[i + 1]; j++) {
				//Transformed coordinates
				int vid = voxel_id[j];
				MatrixDevice<double> centr = centroids(vid);

				double tmp_ex = e_x_cov_x[j];

				if (!(tmp_ex > 1 || tmp_ex < 0 || tmp_ex != tmp_ex)) {
					double cov_dxd0 = cov_dxd_pi_mat0[j];
					double cov_dxd1 = cov_dxd_pi_mat1[j];
					double cov_dxd2 = cov_dxd_pi_mat2[j];

					tmp_ex *= gauss_d1;

					final_hessian += -gauss_d2 * ((d_x - centr(0)) * cov_dxd0 + (d_y - centr(1)) * cov_dxd1 + (d_z - centr(2)) * cov_dxd2) * tmp_hessian(j)(col) * tmp_ex;
					final_hessian += (pg0 * cov_dxd0 + pg1 * cov_dxd1 + pg2 * cov_dxd2) * tmp_ex;
				}
			}

			hessians(i)(row, col) = final_hessian;
		}
	}
}

/* End of adding */

/* Fourth step to compute hessian list */
__global__ void computeHessianListS1(float *trans_x, float *trans_y, float *trans_z,
										int *valid_points,
										int *starting_voxel_id, int *voxel_id, int valid_points_num,
										double *centroid_x, double *centroid_y, double *centroid_z,
										double gauss_d1, double gauss_d2, double *hessians,
										double *e_x_cov_x, double *tmp_hessian, double *cov_dxd_pi,
										double *point_gradients,
										int valid_voxel_num)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	int row = blockIdx.y;
	int col = blockIdx.z;

	if (row < 6 && col < 6) {
		double *cov_dxd_pi_mat0 = cov_dxd_pi + row * valid_voxel_num;
		double *cov_dxd_pi_mat1 = cov_dxd_pi_mat0 + 6 * valid_voxel_num;
		double *cov_dxd_pi_mat2 = cov_dxd_pi_mat1 + 6 * valid_voxel_num;
		double *tmp_h = tmp_hessian + col * valid_voxel_num;
		double *h = hessians + (row * 6 + col) * valid_points_num;
		double *tmp_pg0 = point_gradients + col * valid_points_num;
		double *tmp_pg1 = tmp_pg0 + 6 * valid_points_num;
		double *tmp_pg2 = tmp_pg1 + 6 * valid_points_num;

		for (int i = id; i < valid_points_num; i += stride) {
			int pid = valid_points[i];
			double d_x = static_cast<double>(trans_x[pid]);
			double d_y = static_cast<double>(trans_y[pid]);
			double d_z = static_cast<double>(trans_z[pid]);

			double pg0 = tmp_pg0[i];
			double pg1 = tmp_pg1[i];
			double pg2 = tmp_pg2[i];

			double final_hessian = 0.0;

			for ( int j = starting_voxel_id[i]; j < starting_voxel_id[i + 1]; j++) {
				//Transformed coordinates
				int vid = voxel_id[j];

				double tmp_ex = e_x_cov_x[j];

				if (!(tmp_ex > 1 || tmp_ex < 0 || tmp_ex != tmp_ex)) {
					double cov_dxd0 = cov_dxd_pi_mat0[j];
					double cov_dxd1 = cov_dxd_pi_mat1[j];
					double cov_dxd2 = cov_dxd_pi_mat2[j];

					tmp_ex *= gauss_d1;

					final_hessian += -gauss_d2 * ((d_x - centroid_x[vid]) * cov_dxd0 + (d_y - centroid_y[vid]) * cov_dxd1 + (d_z - centroid_z[vid]) * cov_dxd2) * tmp_h[j] * tmp_ex;
					final_hessian += (pg0 * cov_dxd0 + pg1 * cov_dxd1 + pg2 * cov_dxd2) * tmp_ex;
				}
			}

			h[i] = final_hessian;
		}
	}
}

/* Added on 2019/09/24 */
__global__ void computeHessianListS2V2(MatrixDeviceList<float> trans_cloud,
										int *valid_points,
										int *starting_voxel_id, int *voxel_id, int valid_points_num,
										MatrixDeviceList<double> centroids,
										double gauss_d1, double *e_x_cov_x,
										MatrixDeviceList<double> inverse_covariance,	//3x3
										MatrixDeviceList<double> point_hessians,		//18x6
										MatrixDeviceList<double> hessians,			//6x6
										int valid_voxel_num, int itr_num)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	int row = blockIdx.y;
	int col = blockIdx.z;

	if (row < 6 && col < 6) {
		for (int i = id; i < valid_points_num; i += stride) {
			int pid = valid_points[i];
			VectorR<double> p = trans_cloud(pid);
			VectorR<double> ph = point_hessians(i).col<3>(row * 3, col);
			double final_hessian = hessians(i)(row, col);

			for ( int j = starting_voxel_id[i]; j < starting_voxel_id[i + 1]; j++) {
				//Transformed coordinates
				int vid = voxel_id[j];
				double tmp_ex = e_x_cov_x[j];
				MatrixDevice<double> centr = centroids(vid);
				MatrixDevice<double> icov = inverse_covariance(vid);
				int rows = icov.rows();
				int cols = icov.cols();

				if (!(tmp_ex > 1 || tmp_ex < 0 || tmp_ex != tmp_ex)) {
					tmp_ex *= gauss_d1;

					for (int k = 0; k < rows; k++) {
						double tmp = 0;

						for (int t = 0; t < cols; t++) {
							tmp += icov(k, t) * ph(t);
						}

						final_hessian += (p(k) - centr(k)) * tmp * tmp_ex;
					}
				}
			}

			hessians(i)(row, col) = final_hessian;
		}
	}
}
/* End of adding */

/* Compute sum of a list of matrices */
template <typename Scalar, int Rows, int Cols>
__global__ void matrixSum(Scalar *matrix_list, int full_size, int half_size, int offset)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	int row = blockIdx.y;
	int col = blockIdx.z;

	if (row < Rows && col < Cols) {
		for (int i = index; i < half_size; i += stride) {
			MatrixDevice<Scalar, Rows, Cols> left(offset, matrix_list + i);
			double *right_ptr = (i + half_size < full_size) ? matrix_list + i + half_size : NULL;
			MatrixDevice<Scalar, Rows, Cols> right(offset, right_ptr);

			if (right_ptr != NULL) {
				left(row, col) += right(row, col);
			}
		}
	}
}

/* Compute sum of score_inc list */
__global__ void sumScore(double *score, int full_size, int half_size)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < half_size; i += stride) {
		score[i] += (i + half_size < full_size) ? score[i + half_size] : 0;
	}
}

template <typename PointSourceType, typename PointTargetType>
double GNormalDistributionsTransform<PointSourceType, PointTargetType>::computeDerivatives(Eigen::Matrix<double, 6, 1> &score_gradient,
																							Eigen::Matrix<double, 6, 6> &hessian,
																							MatrixDeviceList<float> trans_cloud,
																							int points_num, Eigen::Matrix<double, 6, 1> pose, bool compute_hessian)
{
	MatrixHost<double> p(6, 1);

	for (int i = 0; i < 6; i++) {
		p(i) = pose(i, 0);
	}

	score_gradient.setZero ();
	hessian.setZero ();

	//Compute Angle Derivatives
	computeAngleDerivatives(p);

	//Radius Search
	int *valid_points, *voxel_id, *starting_voxel_id;
	int valid_voxel_num, valid_points_num;

	valid_points = voxel_id = starting_voxel_id = NULL;

	voxel_grid_.radiusSearch(trans_cloud, points_num, resolution_, INT_MAX, &valid_points, &starting_voxel_id, &voxel_id, &valid_voxel_num, &valid_points_num);

	MatrixDeviceList<double> covariance = voxel_grid_.getCovarianceList();
	MatrixDeviceList<double> inverse_covariance = voxel_grid_.getInverseCovarianceList();
	MatrixDeviceList<double> centroid = voxel_grid_.getCentroidList();
	int *points_per_voxel = voxel_grid_.getPointsPerVoxelList();
	int voxel_num = voxel_grid_.getVoxelNum();

	if (valid_points_num == 0)
		return 0;

	//Update score gradient and hessian matrix

	MatrixDeviceList<double> gradients(6, 1, valid_points_num);
	MatrixDeviceList<double> hessians(6, 6, valid_points_num);
	MatrixDeviceList<double> point_gradients(3, 6, valid_points_num);
	MatrixDeviceList<double> point_hessians(18, 6, valid_points_num);

	double *score;

	checkCudaErrors(cudaMalloc(&score, sizeof(double) * valid_points_num));

	int block_x = (valid_points_num > BLOCK_SIZE_X) ? BLOCK_SIZE_X : valid_points_num;

	int grid_x = (valid_points_num - 1) / block_x + 1;

	dim3 grid;

	std::vector<int2> cell_list = {{1, 3}, {2, 3}, {0, 4}, {1, 4}, {2, 4}, {0, 5}, {1, 5}, {2, 5}};

	int2 *cell_list_dev = NULL;

	checkCudaErrors(cudaMalloc(&cell_list_dev, sizeof(int2) * 32));

	checkCudaErrors(cudaMemcpy(cell_list_dev, cell_list.data(), sizeof(int2) * cell_list.size(), cudaMemcpyHostToDevice));

	computePointGradient<<<grid_x, block_x>>>(source_cloud_, points_number_,
												valid_points, valid_points_num,
												dj_ang_.buffer(),
												point_gradients, cell_list_dev, cell_list.size());
	checkCudaErrors(cudaGetLastError());

	if (compute_hessian) {
		cell_list = std::vector<int2>{{9, 3}, {10, 3}, {11, 3}, {12, 3}, {13, 3}, {14, 3},
										{15, 3}, {16, 3}, {17, 3}, {9, 4}, {10, 4}, {11, 4},
										{12, 4}, {13, 4}, {14, 4}, {15, 4}, {16, 4}, {17, 4},
										{9, 5}, {10, 5}, {11, 5}, {12, 5}, {13, 5}, {14, 5},
										{15, 5}, {16, 5}, {17, 5}};

		checkCudaErrors(cudaMemcpy(cell_list_dev, cell_list.data(), sizeof(int2) * cell_list.size(), cudaMemcpyHostToDevice));

		computePointHessian<<<grid_x, block_x>>>(source_cloud_, points_number_,
												valid_points, valid_points_num,
												dh_ang_.buffer(),
												point_hessians, cell_list_dev, cell_list.size());

		checkCudaErrors(cudaGetLastError());
	}

	checkCudaErrors(cudaDeviceSynchronize());


	MatrixDeviceList<double, 6, 1> tmp_hessian(valid_voxel_num);

	double *e_x_cov_x;

	checkCudaErrors(cudaMalloc(&e_x_cov_x, sizeof(double) * valid_voxel_num));

	double *cov_dxd_pi;

	checkCudaErrors(cudaMalloc(&cov_dxd_pi, sizeof(double) * valid_voxel_num * 3 * 6));

	computeExCovX<<<grid_x, block_x>>>(trans_cloud, valid_points,
										starting_voxel_id, voxel_id, valid_points_num,
										centroid,
										gauss_d1_, gauss_d2_,
										e_x_cov_x,
										inverse_covariance(0, 0), inverse_covariance(0, 1), inverse_covariance(0, 2),
										inverse_covariance(1, 0), inverse_covariance(1, 1), inverse_covariance(1, 2),
										inverse_covariance(2, 0), inverse_covariance(2, 1), inverse_covariance(2, 2));
	checkCudaErrors(cudaGetLastError());

	computeScoreList<<<grid_x, block_x>>>(starting_voxel_id, voxel_id, valid_points_num, e_x_cov_x, gauss_d1_, score);
	checkCudaErrors(cudaGetLastError());

	int block_x2 = (valid_voxel_num > BLOCK_SIZE_X) ? BLOCK_SIZE_X : valid_voxel_num;
	int grid_x2 = (valid_voxel_num - 1) / block_x2 + 1;

	updateExCovX<<<grid_x2, block_x2>>>(e_x_cov_x, gauss_d2_, valid_voxel_num);
	checkCudaErrors(cudaGetLastError());

	grid.x = grid_x;
	grid.y = 3;
	grid.z = 6;

	computeCovDxdPi<<<grid, block_x>>>(valid_points, starting_voxel_id, voxel_id, valid_points_num,
											inverse_covariance(0, 0), voxel_num,
											gauss_d1_, gauss_d2_, point_gradients(0, 0),
											cov_dxd_pi, valid_voxel_num);
	checkCudaErrors(cudaGetLastError());

	grid.x = grid_x;
	grid.y = 6;
	grid.z = 1;

	computeScoreGradientList<<<grid, block_x>>>(trans_cloud, valid_points,
													starting_voxel_id, voxel_id, valid_points_num,
													centroid, voxel_num, e_x_cov_x,
													cov_dxd_pi, gauss_d1_, valid_voxel_num, gradients(0, 0));

	checkCudaErrors(cudaGetLastError());


	if (compute_hessian) {

		grid.y = 6;
		grid.z = 1;


		computeHessianListS0<<<grid, block_x>>>(trans_cloud, valid_points,
												starting_voxel_id, voxel_id, valid_points_num,
												centroid,
												inverse_covariance,
												point_gradients,
												tmp_hessian, valid_voxel_num);
		checkCudaErrors(cudaGetLastError());
		grid.z = 6;

		computeHessianListS1<<<grid, block_x>>>(trans_cloud, valid_points,
													starting_voxel_id, voxel_id, valid_points_num,
													centroid, gauss_d1_, gauss_d2_, hessians,
													e_x_cov_x, tmp_hessian, cov_dxd_pi,
													point_gradients,
													valid_voxel_num);
		checkCudaErrors(cudaGetLastError());

		computeHessianListS2V2<<<grid, block_x>>>(trans_cloud, valid_points,
													starting_voxel_id, voxel_id, valid_points_num,
													centroid, gauss_d1_, e_x_cov_x,
													inverse_covariance,
													point_hessians, hessians, valid_voxel_num, 3);
		checkCudaErrors(cudaGetLastError());

	}

	int full_size = valid_points_num;
	int half_size = (full_size - 1) / 2 + 1;

	while (full_size > 1) {
		block_x = (half_size > BLOCK_SIZE_X) ? BLOCK_SIZE_X : half_size;
		grid_x = (half_size - 1) / block_x + 1;

		grid.x = grid_x;
		grid.y = 6;
		grid.z = 1;
		matrixSum<double, 6, 1><<<grid, block_x>>>(gradients(0, 0), full_size, half_size, valid_points_num);
		checkCudaErrors(cudaGetLastError());

		grid.z = 6;
		matrixSum<double, 6, 6><<<grid, block_x>>>(hessians(0, 0), full_size, half_size, valid_points_num);
		checkCudaErrors(cudaGetLastError());

		sumScore<<<grid_x, block_x>>>(score, full_size, half_size);
		checkCudaErrors(cudaGetLastError());

		full_size = half_size;
		half_size = (full_size - 1) / 2 + 1;
	}

	checkCudaErrors(cudaDeviceSynchronize());

	MatrixDevice<double> dgrad(6, 1, valid_points_num, gradients(0, 0));
	MatrixDevice<double> dhess(6, 6, valid_points_num, hessians(0, 0));
	MatrixHost<double> hgrad(6, 1);
	MatrixHost<double> hhess(6, 6);

	hgrad.moveToHost(dgrad);
	hhess.moveToHost(dhess);

	for (int i = 0; i < 6; i++) {
		score_gradient(i) = hgrad(i);
	}

	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < 6; j++) {
			hessian(i, j) = hhess(i, j);
		}
	}

	double score_inc;

	checkCudaErrors(cudaMemcpy(&score_inc, score, sizeof(double), cudaMemcpyDeviceToHost));

	gradients.free();
	hessians.free();
	point_gradients.free();
	point_hessians.free();

	tmp_hessian.free();

	checkCudaErrors(cudaFree(score));

	checkCudaErrors(cudaFree(e_x_cov_x));
	checkCudaErrors(cudaFree(cov_dxd_pi));

	if (valid_points != NULL)
		checkCudaErrors(cudaFree(valid_points));

	if (voxel_id != NULL)
		checkCudaErrors(cudaFree(voxel_id));

	if (starting_voxel_id != NULL)
		checkCudaErrors(cudaFree(starting_voxel_id));

	return score_inc;
}

template <typename PointSourceType, typename PointTargetType>
void GNormalDistributionsTransform<PointSourceType, PointTargetType>::computeAngleDerivatives(MatrixHost<double> pose, bool compute_hessian)
{
	double cx, cy, cz, sx, sy, sz;

	if (fabs(pose(3)) < 10e-5) {
		cx = 1.0;
		sx = 0.0;
	} else {
		cx = cos(pose(3));
		sx = sin(pose(3));
	}

	if (fabs(pose(4)) < 10e-5) {
		cy = 1.0;
		sy = 0.0;
	} else {
		cy = cos(pose(4));
		sy = sin(pose(4));
	}

	if (fabs(pose(5)) < 10e-5) {
		cz = 1.0;
		sz = 0.0;
	} else {
		cz = cos(pose(5));
		sz = sin(pose(5));
	}


	j_ang_(0) = -sx * sz + cx * sy * cz;
	j_ang_(1) = -sx * cz - cx * sy * sz;
	j_ang_(2) = -cx * cy;

	j_ang_(3) = cx * sz + sx * sy * cz;
	j_ang_(4) = cx * cz - sx * sy * sz;
	j_ang_(5) = -sx * cy;

	j_ang_(6) = -sy * cz;
	j_ang_(7) = sy * sz;
	j_ang_(8) = cy;

	j_ang_(9) = sx * cy * cz;
	j_ang_(10) = -sx * cy * sz;
	j_ang_(11) = sx * sy;

	j_ang_(12) = -cx * cy * cz;
	j_ang_(13) = cx * cy * sz;
	j_ang_(14) = -cx * sy;

	j_ang_(15) = -cy * sz;
	j_ang_(16) = -cy * cz;
	j_ang_(17) = 0;

	j_ang_(18) = cx * cz - sx * sy * sz;
	j_ang_(19) = -cx * sz - sx * sy * cz;
	j_ang_(20) = 0;

	j_ang_(21) = sx * cz + cx * sy * sz;
	j_ang_(22) = cx * sy * cz - sx * sz;
	j_ang_(23) = 0;

	j_ang_.moveToGpu(dj_ang_);

	if (compute_hessian) {

		// h_ang_a2_
		h_ang_(0) = -cx * sz - sx * sy * cz;
		h_ang_(1) = -cx * cz + sx * sy * sz;
		h_ang_(2) = sx * cy;

		// h_ang_a3_
		h_ang_(3) = -sx * sz + cx * sy * cz;
		h_ang_(4) = -cx * sy * sz - sx * cz;
		h_ang_(5) = -cx * cy;

		// h_ang_b2_
		h_ang_(6) = cx * cy * cz;
		h_ang_(7) = -cx * cy * sz;
		h_ang_(8) = cx * sy;

		// h_ang_b3_
		h_ang_(9) = sx * cy * cz;
		h_ang_(10) = -sx * cy * sz;
		h_ang_(11) = sx * sy;

		// h_ang_c2_
		h_ang_(12) = -sx * cz - cx * sy * sz;
		h_ang_(13) = sx * sz - cx * sy * cz;
		h_ang_(14) = 0;

		// h_ang_c3_
		h_ang_(15) = cx * cz - sx * sy * sz;
		h_ang_(16) = -sx * sy * cz - cx * sz;
		h_ang_(17) = 0;

		// h_ang_d1_
		h_ang_(18) = -cy * cz;
		h_ang_(19) = cy * sz;
		h_ang_(20) = sy;

		// h_ang_d2_
		h_ang_(21) = -sx * sy * cz;
		h_ang_(22) = sx * sy * sz;
		h_ang_(23) = sx * cy;

		// h_ang_d3_
		h_ang_(24) = cx * sy * cz;
		h_ang_(25) = -cx * sy * sz;
		h_ang_(26) = -cx * cy;

		// h_ang_e1_
		h_ang_(27) = sy * sz;
		h_ang_(28) = sy * cz;
		h_ang_(29) = 0;

		// h_ang_e2_
		h_ang_(30) = -sx * cy * sz;
		h_ang_(31) = -sx * cy * cz;
		h_ang_(32) = 0;

		// h_ang_e3_
		h_ang_(33) = cx * cy * sz;
		h_ang_(34) = cx * cy * cz;
		h_ang_(35) = 0;

		// h_ang_f1_
		h_ang_(36) = -cy * cz;
		h_ang_(37) = cy * sz;
		h_ang_(38) = 0;

		// h_ang_f2_
		h_ang_(39) = -cx * sz - sx * sy * cz;
		h_ang_(40) = -cx * cz + sx * sy * sz;
		h_ang_(41) = 0;

		// h_ang_f3_
		h_ang_(42) = -sx * sz + cx * sy * cz;
		h_ang_(43) = -cx * sy * sz - sx * cz;
		h_ang_(44) = 0;

		h_ang_.moveToGpu(dh_ang_);
	}

}




__global__ void gpuTransform(MatrixDeviceList<float> input, MatrixDeviceList<float> output,
								int point_num, MatrixDevice<float, 3, 4> transform)	//3x4
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	float x, y, z;

	for (int i = idx; i < point_num; i += stride) {
		MatrixDevice<float> in = input(i);
		MatrixDevice<float> out = output(i);

		x = in(0);
		y = in(1);
		z = in(2);

		out(0) = transform(0, 0) * x + transform(0, 1) * y + transform(0, 2) * z + transform(0, 3);
		out(1) = transform(1, 0) * x + transform(1, 1) * y + transform(1, 2) * z + transform(1, 3);
		out(2) = transform(2, 0) * x + transform(2, 1) * y + transform(2, 2) * z + transform(2, 3);
	}
}

template <typename PointSourceType, typename PointTargetType>
void GNormalDistributionsTransform<PointSourceType, PointTargetType>::transformPointCloud(MatrixDeviceList<float> input,
																							MatrixDeviceList<float> output,
																							int points_number, Eigen::Matrix<float, 4, 4> transform)
{
	Eigen::Transform<float, 3, Eigen::Affine> t(transform);

	MatrixHost<float> htrans(3, 4);
	MatrixDevice<float> dtrans(3, 4);

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 4; j++) {
			htrans(i, j) = t(i, j);
		}
	}

	htrans.moveToGpu(dtrans);

	if (points_number > 0) {
		int block_x = (points_number <= BLOCK_SIZE_X) ? points_number : BLOCK_SIZE_X;
		int grid_x = (points_number - 1) / block_x + 1;

		gpuTransform<<<grid_x, block_x >>>(input, output, points_number, dtrans);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}

	dtrans.free();
}

template <typename PointSourceType, typename PointTargetType>
double GNormalDistributionsTransform<PointSourceType, PointTargetType>::computeStepLengthMT(const Eigen::Matrix<double, 6, 1> &x, Eigen::Matrix<double, 6, 1> &step_dir,
																							double step_init, double step_max, double step_min, double &score,
																							Eigen::Matrix<double, 6, 1> &score_gradient, Eigen::Matrix<double, 6, 6> &hessian,
																							MatrixDeviceList<float> trans_cloud, int points_num)
{
	double phi_0 = -score;
	double d_phi_0 = -(score_gradient.dot(step_dir));

	Eigen::Matrix<double, 6, 1> x_t;

	if (d_phi_0 >= 0) {
		if (d_phi_0 == 0)
			return 0;
		else {
			d_phi_0 *= -1;
			step_dir *= -1;
		}
	}

	int max_step_iterations = 10;
	int step_iterations = 0;


	double mu = 1.e-4;
	double nu = 0.9;
	double a_l = 0, a_u = 0;

	double f_l = auxilaryFunction_PsiMT(a_l, phi_0, phi_0, d_phi_0, mu);
	double g_l = auxilaryFunction_dPsiMT(d_phi_0, d_phi_0, mu);

	double f_u = auxilaryFunction_PsiMT(a_u, phi_0, phi_0, d_phi_0, mu);
	double g_u = auxilaryFunction_dPsiMT(d_phi_0, d_phi_0, mu);

	bool interval_converged = (step_max - step_min) > 0, open_interval = true;

	double a_t = step_init;
	a_t = std::min(a_t, step_max);
	a_t = std::max(a_t, step_min);

	x_t = x + step_dir * a_t;

	final_transformation_ = (Eigen::Translation<float, 3>(static_cast<float>(x_t(0)), static_cast<float>(x_t(1)), static_cast<float>(x_t(2))) *
								Eigen::AngleAxis<float>(static_cast<float>(x_t(3)), Eigen::Vector3f::UnitX()) *
								Eigen::AngleAxis<float>(static_cast<float>(x_t(4)), Eigen::Vector3f::UnitY()) *
								Eigen::AngleAxis<float>(static_cast<float>(x_t(5)), Eigen::Vector3f::UnitZ())).matrix();

	transformPointCloud(source_cloud_, trans_cloud, points_num, final_transformation_);

	score = computeDerivatives(score_gradient, hessian, trans_cloud, points_num, x_t);

	double phi_t = -score;
	double d_phi_t = -(score_gradient.dot(step_dir));
	double psi_t = auxilaryFunction_PsiMT(a_t, phi_t, phi_0, d_phi_0, mu);
	double d_psi_t = auxilaryFunction_dPsiMT(d_phi_t, d_phi_0, mu);

	while (!interval_converged && step_iterations < max_step_iterations && !(psi_t <= 0 && d_phi_t <= -nu * d_phi_0)) {
		if (open_interval) {
			a_t = trialValueSelectionMT(a_l, f_l, g_l, a_u, f_u, g_u, a_t, psi_t, d_psi_t);
		} else {
			a_t = trialValueSelectionMT(a_l, f_l, g_l, a_u, f_u, g_u, a_t, phi_t, d_phi_t);
		}

		a_t = (a_t < step_max) ? a_t : step_max;
		a_t = (a_t > step_min) ? a_t : step_min;

		x_t = x + step_dir * a_t;


		final_transformation_ = (Eigen::Translation<float, 3>(static_cast<float>(x_t(0)), static_cast<float>(x_t(1)), static_cast<float>(x_t(2))) *
								 Eigen::AngleAxis<float>(static_cast<float>(x_t(3)), Eigen::Vector3f::UnitX()) *
								 Eigen::AngleAxis<float>(static_cast<float>(x_t(4)), Eigen::Vector3f::UnitY()) *
								 Eigen::AngleAxis<float>(static_cast<float>(x_t(5)), Eigen::Vector3f::UnitZ())).matrix();

		transformPointCloud(source_cloud_, trans_cloud, points_num, final_transformation_);

		score = computeDerivatives(score_gradient, hessian, trans_cloud, points_num, x_t, false);

		phi_t -= score;
		d_phi_t -= (score_gradient.dot(step_dir));
		psi_t = auxilaryFunction_PsiMT(a_t, phi_t, phi_0, d_phi_0, mu);
		d_psi_t = auxilaryFunction_dPsiMT(d_phi_t, d_phi_0, mu);

		if (open_interval && (psi_t <= 0 && d_psi_t >= 0)) {
			open_interval = false;

			f_l += phi_0 - mu * d_phi_0 * a_l;
			g_l += mu * d_phi_0;

			f_u += phi_0 - mu * d_phi_0 * a_u;
			g_u += mu * d_phi_0;
		}

		if (open_interval) {
			interval_converged = updateIntervalMT(a_l, f_l, g_l, a_u, f_u, g_u, a_t, psi_t, d_psi_t);
		} else {
			interval_converged = updateIntervalMT(a_l, f_l, g_l, a_u, f_u, g_u, a_t, phi_t, d_phi_t);
		}
		step_iterations++;
	}

	if (step_iterations) {
		computeHessian(hessian, trans_cloud, points_num, x_t);
	}

	real_iterations_ += step_iterations;

	return a_t;
}


//Copied from ndt.hpp
template <typename PointSourceType, typename PointTargetType>
double GNormalDistributionsTransform<PointSourceType, PointTargetType>::trialValueSelectionMT (double a_l, double f_l, double g_l,
																								double a_u, double f_u, double g_u,
																								double a_t, double f_t, double g_t)
{
	// Case 1 in Trial Value Selection [More, Thuente 1994]
	if (f_t > f_l) {
		// Calculate the minimizer of the cubic that interpolates f_l, f_t, g_l and g_t
		// Equation 2.4.52 [Sun, Yuan 2006]
		double z = 3 * (f_t - f_l) / (a_t - a_l) - g_t - g_l;
		double w = std::sqrt (z * z - g_t * g_l);
		// Equation 2.4.56 [Sun, Yuan 2006]
		double a_c = a_l + (a_t - a_l) * (w - g_l - z) / (g_t - g_l + 2 * w);

		// Calculate the minimizer of the quadratic that interpolates f_l, f_t and g_l
		// Equation 2.4.2 [Sun, Yuan 2006]
		double a_q = a_l - 0.5 * (a_l - a_t) * g_l / (g_l - (f_l - f_t) / (a_l - a_t));

		if (std::fabs (a_c - a_l) < std::fabs (a_q - a_l))
		  return (a_c);
		else
		  return (0.5 * (a_q + a_c));
	}
	// Case 2 in Trial Value Selection [More, Thuente 1994]
	else if (g_t * g_l < 0) {
		// Calculate the minimizer of the cubic that interpolates f_l, f_t, g_l and g_t
		// Equation 2.4.52 [Sun, Yuan 2006]
		double z = 3 * (f_t - f_l) / (a_t - a_l) - g_t - g_l;
		double w = std::sqrt (z * z - g_t * g_l);
		// Equation 2.4.56 [Sun, Yuan 2006]
		double a_c = a_l + (a_t - a_l) * (w - g_l - z) / (g_t - g_l + 2 * w);

		// Calculate the minimizer of the quadratic that interpolates f_l, g_l and g_t
		// Equation 2.4.5 [Sun, Yuan 2006]
		double a_s = a_l - (a_l - a_t) / (g_l - g_t) * g_l;

		if (std::fabs (a_c - a_t) >= std::fabs (a_s - a_t))
		  return (a_c);
		else
		  return (a_s);
	}
	// Case 3 in Trial Value Selection [More, Thuente 1994]
	else if (std::fabs (g_t) <= std::fabs (g_l)) {
		// Calculate the minimizer of the cubic that interpolates f_l, f_t, g_l and g_t
		// Equation 2.4.52 [Sun, Yuan 2006]
		double z = 3 * (f_t - f_l) / (a_t - a_l) - g_t - g_l;
		double w = std::sqrt (z * z - g_t * g_l);
		double a_c = a_l + (a_t - a_l) * (w - g_l - z) / (g_t - g_l + 2 * w);

		// Calculate the minimizer of the quadratic that interpolates g_l and g_t
		// Equation 2.4.5 [Sun, Yuan 2006]
		double a_s = a_l - (a_l - a_t) / (g_l - g_t) * g_l;

		double a_t_next;

		if (std::fabs (a_c - a_t) < std::fabs (a_s - a_t))
		  a_t_next = a_c;
		else
		  a_t_next = a_s;

		if (a_t > a_l)
		  return (std::min (a_t + 0.66 * (a_u - a_t), a_t_next));
		else
		  return (std::max (a_t + 0.66 * (a_u - a_t), a_t_next));
	}
	// Case 4 in Trial Value Selection [More, Thuente 1994]
	else {
		// Calculate the minimizer of the cubic that interpolates f_u, f_t, g_u and g_t
		// Equation 2.4.52 [Sun, Yuan 2006]
		double z = 3 * (f_t - f_u) / (a_t - a_u) - g_t - g_u;
		double w = std::sqrt (z * z - g_t * g_u);
		// Equation 2.4.56 [Sun, Yuan 2006]
		return (a_u + (a_t - a_u) * (w - g_u - z) / (g_t - g_u + 2 * w));
	}
}

//Copied from ndt.hpp
template <typename PointSourceType, typename PointTargetType>
double GNormalDistributionsTransform<PointSourceType, PointTargetType>::updateIntervalMT (double &a_l, double &f_l, double &g_l,
																							double &a_u, double &f_u, double &g_u,
																							double a_t, double f_t, double g_t)
{
  // Case U1 in Update Algorithm and Case a in Modified Update Algorithm [More, Thuente 1994]
	if (f_t > f_l) {
		a_u = a_t;
		f_u = f_t;
		g_u = g_t;
		return (false);
	}
	// Case U2 in Update Algorithm and Case b in Modified Update Algorithm [More, Thuente 1994]
	else if (g_t * (a_l - a_t) > 0) {
		a_l = a_t;
		f_l = f_t;
		g_l = g_t;
		return (false);
	}
	// Case U3 in Update Algorithm and Case c in Modified Update Algorithm [More, Thuente 1994]
	else if (g_t * (a_l - a_t) < 0) {
		a_u = a_l;
		f_u = f_l;
		g_u = g_l;

		a_l = a_t;
		f_l = f_t;
		g_l = g_t;
		return (false);
	}
	// Interval Converged
	else
		return (true);
}

template <typename PointSourceType, typename PointTargetType>
void GNormalDistributionsTransform<PointSourceType, PointTargetType>::computeHessian(Eigen::Matrix<double, 6, 6> &hessian, MatrixDeviceList<float> trans_cloud, int points_num, Eigen::Matrix<double, 6, 1> &p)
{
	int *valid_points, *voxel_id, *starting_voxel_id;
	int valid_voxel_num, valid_points_num;
	//Radius Search
	voxel_grid_.radiusSearch(trans_cloud, points_num, resolution_, INT_MAX, &valid_points, &starting_voxel_id, &voxel_id, &valid_voxel_num, &valid_points_num);

	MatrixDeviceList<double> centroid = voxel_grid_.getCentroidList();
	MatrixDeviceList<double> covariance = voxel_grid_.getCovarianceList();
	MatrixDeviceList<double> inverse_covariance = voxel_grid_.getInverseCovarianceList();
	int *points_per_voxel = voxel_grid_.getPointsPerVoxelList();
	int voxel_num = voxel_grid_.getVoxelNum();

	if (valid_points_num <= 0)
		return;

	//Update score gradient and hessian matrix
	MatrixDeviceList<double> hessians(6, 6, valid_points_num);
	MatrixDeviceList<double> point_gradients(3, 6, valid_points_num);
	MatrixDeviceList<double> point_hessians(18, 6, valid_points_num);

	int2 *cell_list_dev;
	checkCudaErrors(cudaMalloc(&cell_list_dev, sizeof(int2) * 32));


	int block_x = (valid_points_num > BLOCK_SIZE_X) ? BLOCK_SIZE_X : valid_points_num;
	int grid_x = (valid_points_num - 1) / block_x + 1;
	dim3 grid;

	std::vector<int2> cell_list = {{1, 3}, {2, 3}, {0, 4}, {1, 4}, {2, 4}, {0, 5}, {1, 5}, {2, 5}};

	checkCudaErrors(cudaMemcpy(cell_list_dev, cell_list.data(), sizeof(int2) * cell_list.size(), cudaMemcpyHostToDevice));

	computePointGradient<<<grid_x, block_x>>>(source_cloud_, points_number_,
												valid_points, valid_points_num,
												dj_ang_.buffer(),
												point_gradients,
												cell_list_dev, 8);
	checkCudaErrors(cudaGetLastError());

	cell_list = std::vector<int2>{{9, 3}, {10, 3}, {11, 3}, {12, 3}, {13, 3}, {14, 3},
									{15, 3}, {16, 3}, {17, 3}, {9, 4}, {10, 4}, {11, 4},
									{12, 4}, {13, 4}, {14, 4}, {15, 4}, {16, 4}, {17, 4},
									{9, 5}, {10, 5}, {11, 5}, {12, 5}, {13, 5}, {14, 5},
									{15, 5}, {16, 5}, {17, 5}};

	checkCudaErrors(cudaMemcpy(cell_list_dev, cell_list.data(), sizeof(int2) * cell_list.size(), cudaMemcpyHostToDevice));

	computePointHessian<<<grid_x, block_x>>>(source_cloud_, points_number_,
											valid_points, valid_points_num,
											dh_ang_.buffer(),
											point_hessians, cell_list_dev, cell_list.size());

	checkCudaErrors(cudaGetLastError());

	MatrixDeviceList<double> tmp_hessian(6, 1, valid_voxel_num);

	double *e_x_cov_x;

	checkCudaErrors(cudaMalloc(&e_x_cov_x, sizeof(double) * valid_voxel_num));

	double *cov_dxd_pi;

	checkCudaErrors(cudaMalloc(&cov_dxd_pi, sizeof(double) * valid_voxel_num * 3 * 6));

	computeExCovX<<<grid_x, block_x>>>(trans_cloud, valid_points,
										starting_voxel_id, voxel_id, valid_points_num,
										centroid,
										gauss_d1_, gauss_d2_,
										e_x_cov_x,
										inverse_covariance(0, 0), inverse_covariance(0, 0) + voxel_num, inverse_covariance(0, 0) + 2 * voxel_num,
										inverse_covariance(0, 0) + 3 * voxel_num, inverse_covariance(0, 0) + 4 * voxel_num, inverse_covariance(0, 0) + 5 * voxel_num,
										inverse_covariance(0, 0) + 6 * voxel_num, inverse_covariance(0, 0) + 7 * voxel_num, inverse_covariance(0, 0) + 8 * voxel_num);

	checkCudaErrors(cudaGetLastError());

	grid.x = grid_x;
	grid.y = 3;
	grid.z = 6;
	computeCovDxdPi<<<grid, block_x>>>(valid_points, starting_voxel_id, voxel_id, valid_points_num,
											inverse_covariance(0, 0), voxel_num,
											gauss_d1_, gauss_d2_, point_gradients(0, 0),
											cov_dxd_pi, valid_voxel_num);
	checkCudaErrors(cudaGetLastError());

	int block_x2 = (valid_voxel_num > BLOCK_SIZE_X) ? BLOCK_SIZE_X : valid_voxel_num;
	int grid_x2 = (valid_voxel_num - 1) / block_x2 + 1;


	updateExCovX<<<grid_x2, block_x2>>>(e_x_cov_x, gauss_d2_, valid_voxel_num);
	checkCudaErrors(cudaGetLastError());

	grid.y = 6;
	grid.z = 1;

	computeHessianListS0<<<grid, block_x>>>(trans_cloud, valid_points,
												starting_voxel_id, voxel_id, valid_points_num,
												centroid,
												inverse_covariance,
												point_gradients,
												tmp_hessian, valid_voxel_num);
	checkCudaErrors(cudaGetLastError());

	grid.z = 6;

	computeHessianListS1<<<grid, block_x>>>(trans_cloud, valid_points,
												starting_voxel_id, voxel_id, valid_points_num,
												centroid,
												gauss_d1_, gauss_d2_, hessians,
												e_x_cov_x, tmp_hessian, cov_dxd_pi,
												point_gradients,
												valid_voxel_num);
	checkCudaErrors(cudaGetLastError());

	computeHessianListS2V2<<<grid, block_x>>>(trans_cloud, valid_points,
												starting_voxel_id, voxel_id, valid_points_num,
												centroid, gauss_d1_, e_x_cov_x,
												inverse_covariance,
												point_hessians, hessians, valid_voxel_num, 3);
	checkCudaErrors(cudaGetLastError());


	int full_size = valid_points_num;
	int half_size = (full_size - 1) / 2 + 1;

	while (full_size > 1) {
		block_x = (half_size > BLOCK_SIZE_X) ? BLOCK_SIZE_X : half_size;
		grid_x = (half_size - 1) / block_x + 1;

		grid.x = grid_x;
		grid.y = 6;
		grid.z = 6;
		matrixSum<double, 6, 6><<<grid_x, block_x>>>(hessians(0, 0), full_size, half_size, valid_points_num);

		full_size = half_size;
		half_size = (full_size - 1) / 2 + 1;
	}

	checkCudaErrors(cudaDeviceSynchronize());

	MatrixDevice<double> dhessian(6, 6, valid_points_num, hessians(0, 0));
	MatrixHost<double> hhessian(6, 6);

	hhessian.moveToHost(dhessian);

	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < 6; j++) {
			hessian(i, j) = hhessian(i, j);
		}
	}

	hessians.free();
	point_gradients.free();
	point_hessians.free();
	tmp_hessian.free();

	checkCudaErrors(cudaFree(e_x_cov_x));
	checkCudaErrors(cudaFree(cov_dxd_pi));

	if (valid_points != NULL) {
		checkCudaErrors(cudaFree(valid_points));
	}

	if (voxel_id != NULL) {
		checkCudaErrors(cudaFree(voxel_id));
	}

	if (starting_voxel_id != NULL) {
		checkCudaErrors(cudaFree(starting_voxel_id));
	}

	dhessian.free();
}

template <typename T>
__global__ void gpuSum(T *input, int size, int half_size)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = idx; i < half_size; i += stride) {
		if (i + half_size < size) {
			input[i] += (half_size < size) ? input[i + half_size] : 0;
		}
	}
}

template <typename PointSourceType, typename PointTargetType>
double GNormalDistributionsTransform<PointSourceType, PointTargetType>::getFitnessScore(double max_range)
{
	double fitness_score = 0.0;

	MatrixDeviceList<float> trans_cloud(3, 1, points_number_);

	transformPointCloud(source_cloud_, trans_cloud, points_number_, final_transformation_);

	int *valid_distance;

	checkCudaErrors(cudaMalloc(&valid_distance, sizeof(int) * points_number_));

	double *min_distance;

	checkCudaErrors(cudaMalloc(&min_distance, sizeof(double) * points_number_));

	voxel_grid_.nearestNeighborSearch(trans_cloud, points_number_, valid_distance, min_distance, max_range);

	int size = points_number_;
	int half_size;

	while (size > 1) {
		half_size = (size - 1) / 2 + 1;

		int block_x = (half_size > BLOCK_SIZE_X) ? BLOCK_SIZE_X : half_size;
		int grid_x = (half_size - 1) / block_x + 1;

		gpuSum<double><<<grid_x, block_x>>>(min_distance, size, half_size);
		checkCudaErrors(cudaGetLastError());

		gpuSum<int><<<grid_x, block_x>>>(valid_distance, size, half_size);
		checkCudaErrors(cudaGetLastError());

		size = half_size;
	}

	checkCudaErrors(cudaDeviceSynchronize());

	int nr;

	checkCudaErrors(cudaMemcpy(&nr, valid_distance, sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&fitness_score, min_distance, sizeof(double), cudaMemcpyDeviceToHost));

	trans_cloud.free();
	checkCudaErrors(cudaFree(valid_distance));
	checkCudaErrors(cudaFree(min_distance));

	if (nr > 0)
		return (fitness_score / nr);

	return DBL_MAX;
}


}
