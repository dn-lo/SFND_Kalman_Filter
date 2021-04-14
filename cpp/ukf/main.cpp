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

  // augmentedsigma points 
  MatrixXd Xsig_aug = MatrixXd(7, 15);
  ukf.AugmentedSigmaPoints(&Xsig_aug);
  // print result
  std::cout << "Xsig_aug = " << std::endl << Xsig_aug << std::endl;
  
  return 0;
}