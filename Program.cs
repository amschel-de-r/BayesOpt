using System;
using System.Diagnostics;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using Newtonsoft.Json;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Statistics;

namespace BayesOpt
{
    using Kernels;
    using Utils;
    using Optimisers;
    class Program
    {
        static List<double> nextBest;
        static void Main(string[] args)
        {
            if (args.Length == 0)
            {
                System.Console.WriteLine("No inputs");
                return;
            }
            else if (args.Length > 2)
            {
                System.Console.WriteLine("Too many args");
                return;
            }
            else if (args.Length == 2)
            {
                var a = Matrix<double>.Build.Dense(3, 3, (i,j) => 3*i + j);
                var b = Matrix<double>.Build.Dense(3,3, (i,j) => 3*j + i);
                var V = Vector<double>.Build;
                var c = V.DenseOfArray(new double[]{0, 1, 2, 3, 4});
                var d = c;
                var e = V.DenseOfArray(new double[] {10, 5, 1});
                var white = new RBF();
                var cov = white.Compute(c, d);
                Console.WriteLine(a);
                Console.WriteLine(e);
                Console.WriteLine(a * e);
                
                // testGridSearch(10000);
            }
            else
            {
                int runs;
                if (!int.TryParse(args[0], out runs))
                {
                    System.Console.WriteLine("Number of runs must be an integer");
                    return;
                }
                nextBest = new List<double>();
                run(runs);
            }
        }

        static void testGridSearch(int resolution)
        {
            Console.WriteLine("start");
            GridSearch gs = new GridSearch(
                v => 1 - Math.Pow(v[0],2) - Math.Pow(v[1],2) + 2*v[0] + 4*v[1],
                new double[,]{{-4,4},{-4,4}},
                resolution
            );
            var res = gs.maximise();
            Console.WriteLine(string.Join(',', res.thetaOpt));
        }

        static void run(int runs)
        {
            Func<double, double> func = x => -x * Trig.Cos(-2 * x) * Math.Exp(-x / 3);
            BayesianOptimisation optimizer = new BayesianOptimisation((0, 8), 800, func);
            optimizer.probe(0, lazy: true);
            optimizer.probe(8, lazy: true);
            for (int i = 0; i < runs; i++)
            {
                optimizer.maximise(initPoints: 0, nIter: 1);
                optimizer.suggest();
                logRun(optimizer, i+1);
                System.Console.WriteLine(i);
                // System.Console.WriteLine(optimizer._gp.logMarginalLikelihoodValue);
            }
        }

        static void logRun(BayesianOptimisation optimizer, int run)
        {
            int steps = optimizer.space.Count;
            var res = optimizer.res;
            double[] xObs = res.@params.ToArray();
            double[] yObs = res.target.ToArray();

            double[] x = optimizer.space.ParamSpace;
            double[] mean = optimizer.space.Mean;
            double[] covariance = optimizer.space.Covariance;
            double[] acqVals = optimizer.space.AcquisitionVals;
            double next = optimizer.space.NextBest;
            Console.WriteLine(next);
            nextBest.Add(next);

            var estimationResults = new List<EstimationResult>();
            var queryResults = new List<DataPoint>();
            var aquisitionFunctionValues = new List<AquisitionFunctionValue>();

            xObs.ForEach((i, q) => queryResults.Add(new DataPoint(q, yObs[i])));
            x.ForEach((i, q) => estimationResults.Add(new EstimationResult(mean[i], covariance[i], q)));
            x.ForEach((i, q) => aquisitionFunctionValues.Add(new AquisitionFunctionValue(q, acqVals[i])));

            var er = estimationResults
                .Select(q => new double[] { q.Mean, q.UpperBound, q.LowerBound, q.X })
                .ToArray();

            var qr = queryResults
                .Select(q => new double[] { q.X, q.FX })
                .ToArray();

            var af = aquisitionFunctionValues
                .Select(q => new double[] { q.X, q.FX })
                .ToArray();

            var json1 = JsonConvert.SerializeObject(er, Formatting.Indented);
            string filename1 = "DataOutput/predicted_testCs" + run + ".json";
            File.WriteAllText(filename1, json1);

            var json2 = JsonConvert.SerializeObject(qr, Formatting.Indented);
            string filename2 = "DataOutput/observed_testCs" + run + ".json";
            File.WriteAllText(filename2, json2);

            var json3 = JsonConvert.SerializeObject(af, Formatting.Indented);
            string filename3 = "DataOutput/aquisition_testCs" + run + ".json";
            File.WriteAllText(filename3, json3);

            string filename = "DataOutput/nextbestCs.json";
            var json4 = JsonConvert.SerializeObject(nextBest.ToArray());
            File.WriteAllText(filename, json4);
        }

        static void writeJson(double[] vals, string name, int run)
        {
            var json = JsonConvert.SerializeObject(vals, Formatting.Indented);
            string filename = $"DataOutput/{name}Cs" + run + ".json";
            File.WriteAllText(filename, json);
        }

        static void Time(Action func)
        {
            var w = Stopwatch.StartNew();
            func();
            Console.WriteLine(func.ToString() + ": " + w.Elapsed);
        }

        static void testList()
        {
            Random rng = new Random();
            List<double> xs = new List<double>();
            for (int i = 0; i < 10000; i++)
            {
                xs.Add(rng.NextDouble());
            }
            double av = xs.Average();
        }

        static void testVector()
        {
            Random rng = new Random();
            List<double> xs = new List<double>();
            for (int i = 0; i < 10000; i++)
            {
                xs.Add(rng.NextDouble());
            }
            Vector<double> xsVector = Vector<double>.Build.DenseOfEnumerable(xs);
            double av = xsVector.Mean();
        }
    }

    public class DataPoint
    {
        public readonly double X;
        public readonly double FX;

        public DataPoint(double x, double fx)
        {
            X = x;
            FX = fx;
        }
    }

    public class EstimationResult
    {
        public readonly double Mean;
        public readonly double LowerBound;
        public readonly double UpperBound;
        public readonly double X;

        internal EstimationResult(double mean, double confidence, double x)
        {
            Mean = mean;
            LowerBound = mean - confidence;
            UpperBound = mean + confidence;
            X = x;
        }
    }

    public class AquisitionFunctionValue : IComparable<AquisitionFunctionValue>
    {
        public readonly double X;
        public readonly double FX;

        public AquisitionFunctionValue(double x, double fx)
        {
            X = x;
            FX = fx;
        }

        public int CompareTo(AquisitionFunctionValue other)
        {
            return FX.CompareTo(other.FX);
        }
    }
}
