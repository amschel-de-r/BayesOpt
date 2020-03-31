using System;
using System.Linq;

namespace BayesOpt.Kernels
{
    public class Matern : Kernel
    {
        // TODO inherit from RBF, allow for hyperparameters
        private double lenScale;
        private double nu;
        private static double[] implementedNus = {0.5, 1.5, 2.5};
        public Matern(double lenScale, double nu)
        {
            this.lenScale = lenScale;
            if (!implementedNus.Contains(nu))
            {
                throw new ArgumentException("Matern kernel is currently only implemented for nu = 0.5, 1.5, 2.5");
            }
            this.nu = nu;
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
            double sqDistAdj; // Adjusted

            switch (nu)
            {
                case 0.5:
                    return Math.Exp(-sqDist);

                case 1.5:
                    sqDistAdj = sqDist * Math.Sqrt(3);
                    return (1 + sqDistAdj) * Math.Exp(sqDistAdj);

                case 2.5:
                    sqDistAdj = sqDist * Math.Sqrt(5);
                    return (1 + sqDistAdj + (Math.Pow(sqDistAdj,2)/3) * Math.Exp(-sqDistAdj));
                
                default:
                    return 0; // Other values of nu not yet implemented, exception thrown during constructor
            }
        }
    }
}