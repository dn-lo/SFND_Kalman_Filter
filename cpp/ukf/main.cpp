#include <iostream>
#include "Dense"
#include "ukf.h"

using Eigen::MatrixXd;

int main() {

  // Create a UKF instance
  UKF ukf;

  // sigma points
  MatrixXd Xsig = MatrixXd(5, 11);
  ukf.GenerateSigmaPoints(&Xsig);

  // print result
  std::cout << "Xsig = " << std::endl << Xsig << std::endl;

  // augmented sigma points 
  MatrixXd Xsig_aug = MatrixXd(7, 15);
  ukf.AugmentedSigmaPoints(&Xsig_aug);

  // predict sigma points
  MatrixXd Xsig_pred = MatrixXd(15, 5);
  ukf.SigmaPointPrediction(&Xsig_pred);


  return 0;
}