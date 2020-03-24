using System;
using MathNet.Numerics.LinearAlgebra;

namespace BayesOpt.Kernels
{
    // Radial-basis function kernel (aka squared-exponential kernel)
    public class RBF : Kernel
    {
        private double lenScale;
        public RBF(double lenScale)
        {
            this.lenScale = lenScale;
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