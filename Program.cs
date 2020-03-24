﻿using System;
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
    class Program
    {
        static void Main(string[] args)
        {
            if (args.Length == 0)
            {
                System.Console.WriteLine("No inputs");
                return;
            }
            else if (args.Length > 1)
            {
                System.Console.WriteLine("Too many args");
                return;
            }
            else
            {
                int runs;
                if (!int.TryParse(args[0], out runs))
                {
                    System.Console.WriteLine("Number of runs must be an integer");
                    return;
                }
                run(runs);
            }
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
            double[] nextX = new double[] { optimizer.space.NextBest };

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