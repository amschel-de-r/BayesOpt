using System;
using System.Collections.Generic;
using System.Linq;
using BayesOpt.Utils;
using MathNet.Numerics;

// TODO should optimizer be allowed to probe same point twice?

namespace BayesOpt
{
    public class TargetSpace
    {

        #region Private properties
        private (double min, double max) _bounds; // TODO review with nDim > 1
        private HashSet<(double, double)> _cache;
        private readonly List<double> _params;
        private readonly List<double> _target;
        private double[] _paramSpace;
        private double[] _mean;
        private double[] _covariance;
        private double[] _acqValues;
        private double _nextBest;

        

        #endregion
        
        public TargetSpace((double min, double max) paramBounds, int resolution)
        {
            _bounds = paramBounds;
            _paramSpace = Generate.LinearSpaced(resolution, _bounds.min, _bounds.max);
            _cache = new HashSet<(double, double)>();
            _params = new List<double>();
            _target = new List<double>();
            this.resolution = resolution;
        }

        #region Public properties
        public bool Contains(double x, double y)
        {
            return _cache.Contains((x, y));
        }
        public int Count { get { return _cache.Count; } }
        public List<double> Params { get { return _params; } }
        // TODO is this actually necessary
        public List<double> Target { get { return _target; } }
        public readonly int dim = 1; // TODO with nDim > 1
        public (double, double) bounds { get { return _bounds; } }
        public int resolution;
        public double[] ParamSpace { get { return _paramSpace; } }
        public double[] Mean { get { return _mean; } }
        public double[] Covariance { get { return _covariance; } }
        public double[] AcquisitionVals { get { return _acqValues; } }
        public double NextBest { get { return _nextBest; } }

        #endregion
        
        public void register(double param, double target)
        {
            _params.Add(param);
            _target.Add(target);
            _cache.Add((param, target));
        }

        public double randomSample()
        {
            Random rng = new Random();
            return _bounds.min + rng.NextDouble() * (_bounds.max - _bounds.min);
            // TODO replace with U(min, max)?
        }

        public double max()
        {
            return _target.Max();
        }

        public (List<double> @params, List<double> target) res
        {
            get { return (_params, _target); }
        }

        public void setBounds(double min, double max)
        {
            _bounds.min = min;
            _bounds.max = max;
        }

        public void logOptimisationData(double[] xs, double[] mean, double[] covariance, double[] acqVals, double nextBest)
        {
            _paramSpace = xs;
            _mean = mean;
            _covariance = covariance;
            _acqValues = acqVals;
            _nextBest = nextBest;
        }
    }
}