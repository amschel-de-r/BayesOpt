using System;
using System.Collections.Generic;

namespace BayesOpt
{
    using GPR;
    using Utils;
    using AcquisitionFunctions;
    using Kernels;

    public class BayesianOptimisation
    {
        // rng
        private TargetSpace _space;
        private Func<double, double> targetFunc;
        private List<double> _queue;
        public GaussianProcessRegressor _gp;
        private int resolution;

        public BayesianOptimisation((double min, double max) paramBounds, int resolution, Func<double, double> targetFunc = null)
        {
            _space = new TargetSpace(paramBounds, resolution);
            _gp = new GaussianProcessRegressor(normalizeY: true);
            _queue = new List<double>();
            this.resolution = resolution;
            this.targetFunc = targetFunc;
        }

        #region Public properties
        public TargetSpace space { get { return _space; } }
        // max
        public (List<double> @params, List<double> target) res { get { return _space.res; } }
        #endregion

        public void register(double param, double target)
        {
            _space.register(param, target);
            _gp.fit(param, target); // TODO can this go in suggest?
        }

        public void probe(double param, bool lazy = true)
        {
            if (lazy)
            {
                _queue.Add(param);
            }
            else
            {
                double target = targetFunc(param);
                register(param, target);
            }
        }

        public double suggest(AcquisitionFunction ac = null)
        {
            if (_space.Count == 0)
            {
                return _space.randomSample();
            }

            // TODO bayes_opt fits here, using all previous training data
            
            ac = ac ?? new ExpectedImprovement(_gp, _space);

            return ac.AcqMax();
        }

        private void primeQueue(int initPoints)
        {
            if (_queue.Count == 0 && _space.Count == 0)
            {
                initPoints = Math.Max(initPoints, 1);
            }

            for (int i = 0; i < initPoints; i++)
            {
                _queue.Add(_space.randomSample());
            }
        }
        
        public void maximise(
            int initPoints = 5,
            int nIter = 25,
            double kappa = 2.576,
            double xi = 0
        )
        {
            primeQueue(initPoints);
            // TODO ac options
            // TODO gpr parameters
            AcquisitionFunction ac = new ExpectedImprovement(_gp, _space);
            int iteration = 0;

            while (_queue.Count > 0 || iteration < nIter)
            {
                double x_probe;
                if (_queue.Count != 0)
                {
                    x_probe = _queue.Pop(0);
                }
                else
                {
                    double next = suggest(ac);
                    Console.WriteLine(next);
                    x_probe = next;
                    iteration++;
                }

                probe(x_probe, lazy: false);
            }
        }

        public void setBounds(double min, double max)
        {
            _space.setBounds(min, max);
        }

        public void setParams(
            Kernel kernel = null,
            double? alpha = null,
            bool? normalizeY = null
        )
        {
            _gp.setParams(kernel, alpha, normalizeY);
        }
    }
}