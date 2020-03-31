namespace BayesOpt.Kernels
{
    public class Hyperparameter
    {
        public string name;
        public double value;
        public (double min, double max) bounds;
        public bool isFixed;

        public Hyperparameter(string name, double value,
                              double min = 0, double max = 0,
                              bool isFixed = true)
        {
            this.name = name;
            this.value = value;
            if (min == max)
            {
                isFixed = true;
            }  
            else
            {
                isFixed = false;
                bounds.min = min;
                bounds.max = max;
            }
            this.isFixed = isFixed;
        }
        // TODO does this need an equals method
    }
}