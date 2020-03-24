namespace BayesOpt.Kernels
{
    public class WhiteKernel : Kernel
    {
        private double noiseValue;
        public WhiteKernel(double noiseValue)
        {
            this.noiseValue = noiseValue;
        }
        internal override double Compute(double left, double right) 
        {
            return left == right ? noiseValue : 0;
        }
    }
}