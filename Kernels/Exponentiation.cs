using System;

namespace BayesOpt.Kernels
{
    public class Exponentiation : Kernel
    {
        private Kernel kernel;
        private double exponent;
        public Exponentiation(Kernel kernel, double exponent)
        {
            this.kernel = kernel;
            this.exponent = exponent;
            this.hyperparameters = kernel.hyperparameters;
        }
        internal override double Compute(double left, double right) 
        {
            return Math.Pow(kernel.Compute(left, right), exponent);
        }
    }
}