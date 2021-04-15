#include "ukf.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

UKF::UKF() 
{
  Init();
}

UKF::~UKF() 
{

}

void UKF::Init() 
{

}

void UKF::GenerateSigmaPoints(MatrixXd* Xsig_out) 
{

  // set state dimension
  int n_x = 5;

  // define spreading parameter
  double lambda = 3 - n_x;

  // set example state
  VectorXd x = VectorXd(n_x);
  x <<   5.7441,
         1.3800,
         2.2049,
         0.5015,
         0.3528;

  // set example covariance matrix
  MatrixXd P = MatrixXd(n_x, n_x);
  P <<     0.0043,   -0.0013,    0.0030,   -0.0022,   -0.0020,
          -0.0013,    0.0077,    0.0011,    0.0071,    0.0060,
           0.0030,    0.0011,    0.0054,    0.0007,    0.0008,
          -0.0022,    0.0071,    0.0007,    0.0098,    0.0100,
          -0.0020,    0.0060,    0.0008,    0.0100,    0.0123;

  // create sigma point matrix
  MatrixXd Xsig = MatrixXd(n_x, 2 * n_x + 1);

  // calculate square root of P
  MatrixXd A = P.llt().matrixL();

  // set sigma mean point
  Xsig.col(0) = x;
  // set sigma points to the right and left of mean
  for (unsigned int i=0; i<n_x; i++)
  {
      Xsig.col(i+1)     = x + sqrt(lambda + n_x) * A.col(i); // right
      Xsig.col(i+n_x+1) = x - sqrt(lambda + n_x) * A.col(i); // left
  }

  // print result
  // std::cout << "Xsig = " << std::endl << Xsig << std::endl;

  // write result
  *Xsig_out = Xsig;
}

void UKF::AugmentedSigmaPoints(MatrixXd* Xsig_out) 
{

  // set state dimension
  int n_x = 5;

  // set augmented dimension
  int n_aug = 7;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a = 0.2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd = 0.2;

  // define spreading parameter
  double lambda = 3 - n_aug;

  // set example state
  VectorXd x = VectorXd(n_x);
  x <<   5.7441,
         1.3800,
         2.2049,
         0.5015,
         0.3528;

  // create example covariance matrix
  MatrixXd P = MatrixXd(n_x, n_x);
  P <<     0.0043,   -0.0013,    0.0030,   -0.0022,   -0.0020,
          -0.0013,    0.0077,    0.0011,    0.0071,    0.0060,
           0.0030,    0.0011,    0.0054,    0.0007,    0.0008,
          -0.0022,    0.0071,    0.0007,    0.0098,    0.0100,
          -0.0020,    0.0060,    0.0008,    0.0100,    0.0123;

  // create augmented mean vector
  VectorXd x_aug = VectorXd(7);

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7);

  // create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug, 2 * n_aug + 1);

  // create augmented mean state
  x_aug.head(5) =  x;
  x_aug(5) = 0;
  x_aug(6) = 0;

  // create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P;
  P_aug(5,5) = std_a*std_a;
  P_aug(6,6) = std_yawdd*std_yawdd;

  // create square root matrix
  MatrixXd A_aug = P_aug.llt().matrixL();

  // create augmented sigma points
  // set sigma mean point
  Xsig_aug.col(0) = x_aug;
  // set sigma points to the right and left of mean
  for (unsigned int i=0; i<n_aug; i++)
  {
      Xsig_aug.col(i+1)       = x_aug + sqrt(lambda + n_aug) * A_aug.col(i); // right
      Xsig_aug.col(i+n_aug+1) = x_aug - sqrt(lambda + n_aug) * A_aug.col(i); // left
  }
  // print result
  std::cout << "Xsig_aug = " << std::endl << Xsig_aug << std::endl;

  // write result
  *Xsig_out = Xsig_aug;
}

void UKF::SigmaPointPrediction(MatrixXd* Xsig_out) 
{

  // set state dimension
  int n_x = 5;

  // set augmented dimension
  int n_aug = 7;

  // create example sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug, 2 * n_aug + 1);
  Xsig_aug <<
    5.7441,  5.85768,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.63052,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,
      1.38,  1.34566,  1.52806,     1.38,     1.38,     1.38,     1.38,     1.38,   1.41434,  1.23194,     1.38,     1.38,     1.38,     1.38,     1.38,
    2.2049,  2.28414,  2.24557,  2.29582,   2.2049,   2.2049,   2.2049,   2.2049,   2.12566,  2.16423,  2.11398,   2.2049,   2.2049,   2.2049,   2.2049,
    0.5015,  0.44339, 0.631886, 0.516923, 0.595227,   0.5015,   0.5015,   0.5015,   0.55961, 0.371114, 0.486077, 0.407773,   0.5015,   0.5015,   0.5015,
    0.3528, 0.299973, 0.462123, 0.376339,  0.48417, 0.418721,   0.3528,   0.3528,  0.405627, 0.243477, 0.329261,  0.22143, 0.286879,   0.3528,   0.3528,
         0,        0,        0,        0,        0,        0,  0.34641,        0,         0,        0,        0,        0,        0, -0.34641,        0,
         0,        0,        0,        0,        0,        0,        0,  0.34641,         0,        0,        0,        0,        0,        0, -0.34641;

  // create matrix with predicted sigma points as columns
  MatrixXd Xsig_pred = MatrixXd(n_x, 2 * n_aug + 1);

  double dt = 0.1; // time diff in sec

  // predict sigma points
  for (unsigned int i = 0; i<(2*n_aug+1); i++)
  {
    // get state variables and noises from sigma point 
    double x        = Xsig_aug(0, i);
    double y        = Xsig_aug(1, i);    
    double v        = Xsig_aug(2, i);
    double psi      = Xsig_aug(3, i);
    double psid     = Xsig_aug(4, i);
    double nu_a     = Xsig_aug(5, i);
    double nu_psidd = Xsig_aug(6, i);

    // initialize new state variables
    double x_n, y_n, v_n, psi_n, psid_n;

    // if yaw rate is not zero
    if (psid != 0)
    {
      // increment x and y using normal formula
      x_n = x + v / psid * (sin(psi + psid * dt) - sin(psi));
      y_n = y + v / psid * (-cos(psi + psid * dt) + cos(psi));
    }
    else
    {
      // avoid division by zero
      x_n = v * cos(psi) * dt;
      y_n = v * sin(psi) * dt;
    }
    // increment remaining state variables
    v_n    = v;
    psi_n  = psi + psid * dt;
    psid_n = psid;

    // add noise
    double  dt2 = dt*dt;
    x_n     = x_n + 0.5 * dt2 * cos(psi) * nu_a;
    y_n     = y_n + 0.5 * dt2 * sin(psi) * nu_a;
    v_n     = v_n + dt * nu_a;
    psi_n   = psi_n + 0.5 * dt2 * nu_psidd;
    psid_n  = psid_n + dt * nu_psidd;

    // write predicted sigma points into right column
    Xsig_pred(0,i) = x_n;
    Xsig_pred(1,i) = y_n;
    Xsig_pred(2,i) = v_n;
    Xsig_pred(3,i) = psi_n;
    Xsig_pred(4,i) = psid_n;
  }

  // print result
  std::cout << "Xsig_pred = " << std::endl << Xsig_pred << std::endl;

  // write result
  *Xsig_out = Xsig_pred;
}

void UKF::PredictMeanAndCovariance(VectorXd* x_out, MatrixXd* P_out) 
{

  // set state dimension
  int n_x = 5;

  // set augmented dimension
  int n_aug = 7;

  // define spreading parameter
  double lambda = 3 - n_aug;

  // create example matrix with predicted sigma points
  MatrixXd Xsig_pred = MatrixXd(n_x, 2 * n_aug + 1);
  Xsig_pred <<
         5.9374,  6.0640,   5.925,  5.9436,  5.9266,  5.9374,  5.9389,  5.9374,  5.8106,  5.9457,  5.9310,  5.9465,  5.9374,  5.9359,  5.93744,
           1.48,  1.4436,   1.660,  1.4934,  1.5036,    1.48,  1.4868,    1.48,  1.5271,  1.3104,  1.4787,  1.4674,    1.48,  1.4851,    1.486,
          2.204,  2.2841,  2.2455,  2.2958,   2.204,   2.204,  2.2395,   2.204,  2.1256,  2.1642,  2.1139,   2.204,   2.204,  2.1702,   2.2049,
         0.5367, 0.47338, 0.67809, 0.55455, 0.64364, 0.54337,  0.5367, 0.53851, 0.60017, 0.39546, 0.51900, 0.42991, 0.530188,  0.5367, 0.535048,
          0.352, 0.29997, 0.46212, 0.37633,  0.4841, 0.41872,   0.352, 0.38744, 0.40562, 0.24347, 0.32926,  0.2214, 0.28687,   0.352, 0.318159;

  // create vector for weights
  VectorXd weights = VectorXd(2*n_aug+1);
  
  // create vector for predicted state
  VectorXd x = VectorXd(n_x);

  // create covariance matrix for prediction
  MatrixXd P = MatrixXd(n_x, n_x);

  // set weights
  weights.fill(0.5/(lambda+n_aug));
  weights(0) = lambda / (lambda + n_aug);

  // predict state mean
  x = Xsig_pred * weights;

  // predict state covariance matrix
  P.fill(0.0);
  // iterate over sigma points
  for (unsigned int i = 0; i < 2*n_aug+1; i++)
  {
    VectorXd x_diff = Xsig_pred.col(i) - x;
    // normalize angle to -pi, pi
    while (x_diff(3) > M_PI)
    {
      x_diff(3) -= 2*M_PI;
    }  
    while (x_diff(3) < -M_PI)
    {
      x_diff(3) += 2*M_PI;
    }
    P += weights(i) * x_diff * x_diff.transpose();
  }
  // print result
  std::cout << "Predicted state" << std::endl;
  std::cout << x << std::endl;
  std::cout << "Predicted covariance matrix" << std::endl;
  std::cout << P << std::endl;

  // write result
  *x_out = x;
  *P_out = P;
}

void UKF::PredictRadarMeasurement(VectorXd* z_out, MatrixXd* S_out) 
{

  // set state dimension
  int n_x = 5;

  // set augmented dimension
  int n_aug = 7;

  // set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  // define spreading parameter
  double lambda = 3 - n_aug;

  // set vector for weights
  VectorXd weights = VectorXd(2*n_aug+1);
  weights.fill(0.5/(lambda+n_aug));
  weights(0) = lambda/(lambda+n_aug);

  // radar measurement noise standard deviation radius in m
  double std_radr = 0.3;

  // radar measurement noise standard deviation angle in rad
  double std_radphi = 0.0175;

  // radar measurement noise standard deviation radius change in m/s
  double std_radrd = 0.1;

  // create example matrix with predicted sigma points
  MatrixXd Xsig_pred = MatrixXd(n_x, 2 * n_aug + 1);
  Xsig_pred <<
         5.9374,  6.0640,   5.925,  5.9436,  5.9266,  5.9374,  5.9389,  5.9374,  5.8106,  5.9457,  5.9310,  5.9465,  5.9374,  5.9359,  5.93744,
           1.48,  1.4436,   1.660,  1.4934,  1.5036,    1.48,  1.4868,    1.48,  1.5271,  1.3104,  1.4787,  1.4674,    1.48,  1.4851,    1.486,
          2.204,  2.2841,  2.2455,  2.2958,   2.204,   2.204,  2.2395,   2.204,  2.1256,  2.1642,  2.1139,   2.204,   2.204,  2.1702,   2.2049,
         0.5367, 0.47338, 0.67809, 0.55455, 0.64364, 0.54337,  0.5367, 0.53851, 0.60017, 0.39546, 0.51900, 0.42991, 0.530188,  0.5367, 0.535048,
          0.352, 0.29997, 0.46212, 0.37633,  0.4841, 0.41872,   0.352, 0.38744, 0.40562, 0.24347, 0.32926,  0.2214, 0.28687,   0.352, 0.318159;

  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug + 1);

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  
  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);

  // Transform sigma points into measurement space
  for (unsigned int i = 0; i<(2*n_aug+1); i++)
  {
    // get state variables from sigma point 
    double x        = Xsig_pred(0, i);
    double y        = Xsig_pred(1, i);    
    double v        = Xsig_pred(2, i);
    double psi      = Xsig_pred(3, i);

    // measurement model
    double rho  = sqrt(x*x + y*y);
    double phi  = atan2(y, x); 
    double rhod = (x * cos(psi) * v + y * sin(psi) * v) / rho;

    // assign to sigma point matrix
    Zsig.col(i) << rho, phi, rhod;
  }
  // calculate mean predicted measurement
  z_pred = Zsig * weights;
  
  // calculate innovation covariance matrix S
  S.fill(0.0);
  // iterate over sigma points
  for (unsigned int i = 0; i < 2*n_aug+1; i++)
  {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    
    // normalize phi angles to -pi, pi not needed as they are obtained from atan2
    while (z_diff(1) > M_PI)
    {
      z_diff(1) -= 2*M_PI;
    }  
    while (z_diff(1) < -M_PI)
    {
      z_diff(1) += 2*M_PI;
    }
    
    S += weights(i) * z_diff * z_diff.transpose();
  }
  // measurement noise
  MatrixXd R = MatrixXd(n_z,n_z);
  R <<  std_radr * std_radr,  0,  0,
        0,  std_radphi * std_radphi,  0,
        0,  0,  std_radrd * std_radrd;
  
  // add noise to covariance 
  S = S + R;

  // print result
  std::cout << "z_pred: " << std::endl << z_pred << std::endl;
  std::cout << "S: " << std::endl << S << std::endl;

  // write result
  *z_out = z_pred;
  *S_out = S;
}