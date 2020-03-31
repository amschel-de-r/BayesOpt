using System;

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
            int l1 = k1.hyperparameters.Length;
            int l2 = k2.hyperparameters.Length;
            hyperparameters = new Hyperparameter[l1 + l2];
            Array.Copy(k1.hyperparameters, hyperparameters, l1);
            Array.Copy(k2.hyperparameters, 0, hyperparameters, l1, l2);
        }
    }
}