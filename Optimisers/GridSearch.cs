using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics;

namespace BayesOpt.Optimisers
{
    using Kernels;
    using Utils;
    public class GridSearch
    {
        private Func<double[], double> func;
        private double[,] bounds;
        private int resolution;
        public GridSearch(Func<double[], double> func, double[,] bounds, int resolution)
        {
            if (bounds.Rank != 2)
                throw new ArgumentException("Bounds must be 2-dimensional");
            
            if (bounds.GetLength(1) != 2)
                throw new ArgumentException("Second dimension of bounds must be length 2: min, max");

            this.func = func;
            this.bounds = bounds;
            this.resolution = resolution;
        }
        public (double[] thetaOpt, double funcResult) maximise()
        {
            List<IEnumerator> enumerators = new List<IEnumerator>();
            List<(double[], double)> results = new List<(double[], double)>();
            for (int i = 0; i < bounds.GetLength(0); i++)
            {
                IEnumerator enumerator = Generate.LinearSpaced(resolution, bounds[i,0], bounds[i,1]).GetEnumerator();
                enumerator.MoveNext();
                enumerators.Add(enumerator);
            }

            bool finished = false;
            enumerators[0].Reset();

            while (true)
            {
                foreach (IEnumerator enumerator in enumerators)
                {
                    if (enumerator.MoveNext())
                    {
                        finished = false;
                        break;
                    }
                    else
                    {
                        enumerator.Reset();
                        enumerator.MoveNext();
                        finished = true;
                    }
                }
                if (finished)
                    break;
                
                double[] thetaCombo = enumerators.ConvertAll(q => (double) q.Current).ToArray();
                results.Add((thetaCombo, func(thetaCombo)));
            }
            
            return results[results.ArgMax(t => t.Item2)];
        }


    }
}