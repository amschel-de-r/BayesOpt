namespace BayesOpt.Kernels
{
    public class Product : KernelOperator
    {
        public Product(Kernel k1, Kernel k2) : base(k1, k2) {}
        internal override double Compute(double left, double right) 
        {
            return k1.Compute(left, right) * k2.Compute(left, right);
        }
    }
}