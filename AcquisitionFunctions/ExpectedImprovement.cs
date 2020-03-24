using System;
using MathNet.Numerics.Distributions;

namespace BayesOpt.AcquisitionFunctions
{
    using GPR;
    internal class ExpectedImprovement : AcquisitionFunction
    {
        private double xi;
        private Normal norm;

        public ExpectedImprovement(
            GaussianProcessRegressor gp, (double min, double max) bounds, int resolution,
            double xi = 0
        ) : base(gp, bounds, resolution)
        {
            init(xi);
        }

        public ExpectedImprovement(
            GaussianProcessRegressor gp, TargetSpace ts,
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
            var res = predict(x);
            double mean = res.mean;
            double std = Math.Sqrt(res.covariance);
            double yMax = this.yMax;

            double z = (mean - yMax - xi) / std;
            return (mean - yMax - xi) * norm.CumulativeDistribution(z) + std * norm.Density(z);
        }
    }
}