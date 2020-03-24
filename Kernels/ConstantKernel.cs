namespace BayesOpt.Kernels
{
    public class ConstantKernel : Kernel
    {
        private double constantValue;
        public ConstantKernel(double constantValue)
        {
            this.constantValue = constantValue;
        }
        internal override double Compute(double left, double right) 
        {
            return constantValue;
        }
    }
}