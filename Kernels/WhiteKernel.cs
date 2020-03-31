namespace BayesOpt.Kernels
{
    public class WhiteKernel : Kernel
    {
        private double noiseValue { get { return hyperparameters[0].value; } }
        public WhiteKernel(double noiseValue = 1, double noiseValueMin = 1e-5, double noiseValueMax = 1e5)
        {
            hyperparameters = new Hyperparameter[1];
            hyperparameters[0] = new Hyperparameter("noiseValue", noiseValue, noiseValueMin, noiseValueMax);
        }
        internal override double Compute(double left, double right) 
        {
            return left == right ? noiseValue : 0;
        }
    }
}