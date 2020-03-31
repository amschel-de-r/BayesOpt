using System;
using MathNet.Numerics.Distributions;

namespace BayesOpt.AcquisitionFunctions
{
    using GPR;
    internal class ProbabilityOfImprovement : AcquisitionFunction
    {
        private double xi;
        private Normal norm;

        public ProbabilityOfImprovement(
            GaussianProcessRegressor gp,
            (double min, double max) bounds,
            int resolution,
            double xi = 0
        ) : base(gp, bounds, resolution)
        {
            init(xi);
        }

        public ProbabilityOfImprovement(
            GaussianProcessRegressor gp,
            TargetSpace ts,
            double xi = 0
        ) : base(gp, ts)
        {
            init(xi);
        }

        private void init(double xi)
        {
            this.xi = xi;
            norm = new Normal();
        }

        public override double AcqValue(double x)
        {
            // TODO compare to koryakinp and rob
            var res = predict(x, returnStd: true);
            double mean = res.mean;
            double std = res.covariance;
            double yMax = this.yMax;

            double z = (mean - yMax - xi) / std;
            return norm.CumulativeDistribution(z);
        }
    }
}