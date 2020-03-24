using BayesOpt.GPR;
using BayesOpt.Utils;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.Random;

namespace BayesOpt.AcquisitionFunctions
{
    public abstract class AcquisitionFunction
    {
        protected GaussianProcessRegressor _gp;
        private (double min, double max) _bounds;
        private int resolution;
        private TargetSpace targetSpace;
        protected double yMax { get { return targetSpace.max(); } }
        private List<double> mean;
        private List<double> covar;
        private bool logData = false;

        public AcquisitionFunction(GaussianProcessRegressor gp, (double min, double max) bounds, int resolution)
        {
            _gp = gp;
            _bounds = bounds;
            this.resolution = resolution;
        }

        public AcquisitionFunction(GaussianProcessRegressor gp, TargetSpace ts) : this(gp, ts.bounds, ts.resolution)
        {
            this.targetSpace = ts;
            logData = true;
        }
        // TODO tidy up constructors

        public abstract double AcqValue(double x);

        public double[] AcqValue(double[] xs)
        {
            double[] vals = new double[xs.Length];
            vals.ForEach((i, _) => AcqValue(xs[i]));
            return vals;
        }

        protected (double mean, double covariance) predict(double x)
        {
            var prediction = _gp.predict(x);
            if (logData)
            {
                mean.Add(prediction.mean);
                covar.Add(prediction.covariance);
            }
            return prediction;
        }

        public double AcqMax()
        {
            ContinuousUniform uniform = new ContinuousUniform(_bounds.min, _bounds.max);
            double[] xTries = new double[resolution];
            uniform.Samples(xTries);
            xTries = xTries.OrderBy(q => q).ToArray();
            mean = new List<double>();
            covar = new List<double>();

            double[] ys = AcqValue(xTries);
            double xMax = xTries[ys.ArgMax()];

            if (logData)
            {
                targetSpace.logOptimisationData(
                    xTries,
                    mean.ToArray(),
                    covar.ToArray(),
                    ys,
                    xMax
                );
            }

            return xMax;
        }
    }
}