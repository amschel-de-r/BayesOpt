using MathNet.Numerics.LinearAlgebra;

namespace BayesOpt.Kernels
{
    public abstract class Kernel
    {
        internal abstract double Compute(double left, double right);

        internal double Compute(double x) { return Compute(x, x); }

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