namespace BayesOpt.Kernels
{
    public abstract class KernelOperator : Kernel
    {
        protected Kernel k1;
        protected Kernel k2;
        protected KernelOperator(Kernel k1, Kernel k2)
        {
            this.k1 = k1;
            this.k2 = k2;
        }
    }
}