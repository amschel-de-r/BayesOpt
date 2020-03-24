using BayesOpt.Kernels;
using BayesOpt.Utils;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Factorization;
using System;
using System.Collections.Generic;
using System.Linq;

namespace BayesOpt.GPR
{
    public class GaussianProcessRegressor
    {
        private Kernel kernel;
        private Matrix<double> covariance;
        private double alpha;
        // optimizer
        // nRestartsOptimizer
        private bool normalizeY;
        // random state (int?)
        private List<double> xTrain;
        private List<double> yTrain;
        private double yTrainMean;
        private Vector<double> yTrainNormalized;
        private Cholesky<double> cho;
        private Vector<double> alpha_;
        // logMargLike
        public GaussianProcessRegressor(Kernel kernel = null, bool normalizeY = false, double alpha = 1e-3)
        {
            // TEST speed
            this.kernel = kernel ?? new RBF(lenScale: 1) + new WhiteKernel(noiseValue: alpha); // default kernel if none supplied
            this.normalizeY = normalizeY;
            this.alpha = alpha;
            xTrain = new List<double>();
            yTrain = new List<double>();
            covariance = Matrix<double>.Build.Dense(1, 1);
        }

        public void fit(double x, double y)
        {
            xTrain.Add(x);
            yTrain.Add(y);
            if (normalizeY)
            {
                yTrainMean = yTrain.Average();
                yTrainNormalized = Vector<double>.Build.DenseOfArray(yTrain.ToArray()) - yTrainMean;
            }
            else
            {
                yTrainNormalized = Vector<double>.Build.DenseOfArray(yTrain.ToArray());
            }

            updateCovariance(x);

            try
            {
                cho = covariance.Cholesky();
            }
            catch (System.ArgumentException)
            {
                throw new System.ArgumentException(
                    ("The kernel is not returning a positive definite matrix. " +
                     "Try gradually increasing the 'alpha' parameter of your GaussianProcessRegressor estimator.")
                );
            }
            
            alpha_ = cho.Solve(yTrainNormalized);
        }

        public (double mean, double covariance) predict(double xQuery)
        {
            if (xTrain.Count == 0)
            {
                return (mean: 0, covariance: kernel.Compute(xQuery));
            }

            double[] kstarArray = xTrain.ToArray();
            kstarArray.ForEach(x => kernel.Compute(xQuery, x));
            var kstar = Vector<double>.Build.DenseOfArray(kstarArray);

            double mean = kstar.DotProduct(alpha_);
            mean = normalizeY ? mean + yTrainMean : mean;

            var v = cho.Solve(kstar);
            double covariance = kernel.Compute(xQuery, xQuery) - kstar.DotProduct(v);
            
            return (mean, covariance);
        }

        public (double[] mean, double[] covariance) predict(double[] xQueries)
        {
            double[] mean = new double[xQueries.Length];
            double[] covariance = new double[xQueries.Length];

            for (int i = 0; i < xQueries.Length; i++)
            {
                var res = predict(xQueries[i]);
                mean[i] = res.mean;
                covariance[i] = res.covariance;
            }

            return (mean, covariance);
        }

        private void updateCovariance(double xNew)
        {
            int size = xTrain.Count;
            var updated = Matrix<double>.Build.Dense(size, size);
            covariance.ForEach((i, j, q) => updated[i, j] = q);

            for (int i = 0; i < size - 1; i++)
            {
                var value = kernel.Compute(xTrain[i], xNew);
                updated[i, size - 1] = value;
                updated[size - 1, i] = value;
            }

            updated[size - 1, size - 1] = kernel.Compute(xNew);
            updated.MapInplace(q => Math.Round(q, 5));
            covariance = updated;
        }

        // sampleY(...)

        // logMargLike(...)

        // constrainedOptimization...
    }
}