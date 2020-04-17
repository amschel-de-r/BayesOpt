using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.Linq;
using System;

namespace BayesOpt.Kernels
{
    public abstract class Kernel
    {
        internal Hyperparameter[] hyperparameters;
        public double[] theta 
        { 
            get 
            {
                if (hyperparameters == null || hyperparameters.Length == 0)
                {
                    return new double[0];
                }

                var nonFixed = hyperparameters.Where(h => !h.isFixed);
                return nonFixed.Select(h => Math.Log(h.value)).ToArray();
            }
            set
            {
                int i = 0;
                foreach (Hyperparameter h in hyperparameters)
                {
                    if (h.isFixed)
                        continue;
                    
                    h.value = Math.Exp(value[i]);
                    i++;
                }
            }
        }
        public double[,] bounds
        {
            get
            {
                if (hyperparameters == null || hyperparameters.Length == 0)
                {
                    return new double[0,0];
                }

                var nonFixed = hyperparameters.Where(h => !h.isFixed)
                                              .Select((h,i) => new{i,h.bounds});
                double[,] b = new double[nonFixed.Count(), 2];
                foreach (var h in nonFixed)
                {
                    b[h.i,0] = h.bounds.min;
                    b[h.i,1] = h.bounds.max;
                }
                return b;
            }
        }
        MatrixBuilder<double> M = Matrix<double>.Build;
        VectorBuilder<double> V = Vector<double>.Build;
        internal abstract double Compute(double left, double right);
        internal double Compute(double x) { return Compute(x, x); }
        public Matrix<double> Compute(Vector<double> X, Vector<double> Y = null)
        {
            Y = Y ?? X;
            var matX = M.DenseOfColumnVectors(Enumerable.Repeat(X, Y.Count));
            var matY = M.DenseOfRowVectors(Enumerable.Repeat(Y, X.Count));
            var result = M.Dense(X.Count, Y.Count);
            matX.Map2(Compute, matY, result, Zeros.Include);
            return result;
        }

        public Matrix<double> Compute(IEnumerable<double> X, IEnumerable<double> Y = null)
        {
            var vX = X as Vector<double>;
            if (vX == null)
                vX = V.DenseOfEnumerable(X);
                
            Y = Y ?? X;

            var vY = Y as Vector<double>;
            if (vY == null)
                vY = V.DenseOfEnumerable(Y);

            return Compute(vX, vY);
        }
        
        // TODO compare speed with [,] array

        public static implicit operator Kernel(double constant)
        {
            return new ConstantKernel(constant);
        }
        public static Kernel operator +(Kernel k1, Kernel k2)
        {
            return new Sum(k1, k2);
        }

        public static Kernel operator -(Kernel kernel)
        {
            return new Product(kernel, -1);
        }

        public static Kernel operator -(Kernel k1, Kernel k2)
        {
            return new Sum(k1, -k2);
        }

        public static Kernel operator *(Kernel k1, Kernel k2)
        {
            return new Product(k1, k2);
        }

        public static Kernel operator ^(Kernel kernel, double exponent)
        {
            return new Exponentiation(kernel, exponent);
        }
    }
}