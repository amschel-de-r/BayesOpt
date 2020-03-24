using System;

namespace BayesOpt.AcquisitionFunctions
{
    using GPR;
    internal class UpperConfidenceBound : AcquisitionFunction
    {
        private double kappa;

        public UpperConfidenceBound(
            GaussianProcessRegressor gp,
            (double min, double max) bounds,
            int resolution,
            double kappa = 2.576
        ) : base(gp, bounds, resolution)
        {
            this.kappa = kappa;
        }

        public UpperConfidenceBound(
            GaussianProcessRegressor gp,
            TargetSpace ts,
            double kappa = 2.576
        ) : base(gp, ts)
        {
            this.kappa = kappa;
        }

        public override double AcqValue(double x)
        {
            // TODO compare to koryakinp and rob
            var res = predict(x); 
            double mean = res.mean;
            double std = Math.Sqrt(res.covariance);

            return mean + kappa * std;
        }
    }
}