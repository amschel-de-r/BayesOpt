using System;
using MathNet.Numerics.LinearAlgebra;

namespace BayesOpt.Kernels
{
    // Radial-basis function kernel (aka squared-exponential kernel)
    public class RBF : Kernel
    {
        private double lenScale { get { return hyperparameters[0].value; } }
        public RBF(double lenScale = 1, double lenScaleMin = 1e-5, double lenScaleMax = 1e5)
        {
            hyperparameters = new Hyperparameter[1];
            hyperparameters[0] = new Hyperparameter("lenScale", lenScale, lenScaleMin, lenScaleMax);
        }
        internal override double Compute(double left, double right) 
        {
            // K(xi, xj) = exp(-1/2((xi - xj)/lenscale)^2)
            if(left == right)
            {
                return 1;
            }

            double diff = left - right;
            double sqDist = Math.Pow((diff / lenScale), 2);
            return Math.Exp(-0.5 * sqDist);
        }
    }
}