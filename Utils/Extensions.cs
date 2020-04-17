using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;

namespace BayesOpt.Utils
{
    internal static class Extensions
    {
        public static void ForEach(this Matrix<double> matrix, Action<int,int,double> action)
        {
            for (int i = 0; i < matrix.RowCount; i++)
            {
                for (int j = 0; j < matrix.ColumnCount; j++)
                {
                    action(i, j, matrix[i, j]);
                }
            }
        }

        public static void ForEach<T>(this T[] input, Action<T> action)
        {
            for (int i = 0; i < input.Length; i++)
            {
                action(input[i]);
            }
        }

        public static void ForEach<T>(this T[] input, Func<T,T> func)
        {
            for (int i = 0; i < input.Length; i++)
            {
                input[i] = func(input[i]);
            }
        }

        public static void ForEach<T>(this T[] input, Action<int, T> action)
        {
            for (int i = 0; i < input.Length; i++)
            {
                action(i, input[i]);
            }
        }

        public static void ForEach<T>(this T[] input, Func<int, T, T> func)
        {
            for (int i = 0; i < input.Length; i++)
            {
                input[i] = func(i, input[i]);
            }
        }

        public static void ForEach<T>(this T[,] input, Action<T> action)
        {
            for (int i = 0; i < input.GetLength(0); i++)
            {
                for (int j = 0; j < input.GetLength(1); j++)
                {
                    action(input[i, j]);
                }
            }
        }

        public static void ForEach<T>(this T[,] input, Action<int,int,T> action)
        {
            for (int i = 0; i < input.GetLength(0); i++)
            {
                for (int j = 0; j < input.GetLength(1); j++)
                {
                    action(i, j, input[i, j]);
                }
            }
        }

        public static void ForEach<T>(this T[,] input, Action<int, int> action)
        {
            for (int i = 0; i < input.GetLength(0); i++)
            {
                for (int j = 0; j < input.GetLength(1); j++)
                {
                    action(i, j);
                }
            }
        }

        public static TSrc ArgMaxWeird<TSrc, TArg>(this IEnumerable<TSrc> ie, Converter<TSrc, TArg> fn)
            where TArg : IComparable<TArg>
        {
            IEnumerator<TSrc> e = ie.GetEnumerator();
            if (!e.MoveNext())
                throw new InvalidOperationException("Sequence has no elements.");

            TSrc t_try, t = e.Current;
            if (!e.MoveNext())
                return t;

            TArg v, max_val = fn(t);
            do
            {
                if ((v = fn(t_try = e.Current)).CompareTo(max_val) > 0)
                {
                    t = t_try;
                    max_val = v;
                }
            }
            while (e.MoveNext());
            return t;
        }

        public static int ArgMax<T>(this IEnumerable<T> ie)
            where T : IComparable<T>
        {
            IEnumerator<T> e = ie.GetEnumerator();
            if (!e.MoveNext())
                return -1;

            int max_ix = 0;
            T t_try = e.Current;
            if (!e.MoveNext())
                return max_ix;

            T tx, max_val = e.Current;
            int i = 1;
            do
            {
                if ((tx = e.Current).CompareTo(max_val) > 0)
                {
                    max_val = tx;
                    max_ix = i;
                }
                i++;
            }
            while (e.MoveNext());
            return max_ix;
        }

        public static int ArgMax<TSrc, TArg>(this IEnumerable<TSrc> ie, Converter<TSrc, TArg> fn)
            where TArg : IComparable<TArg>
        {
            IEnumerator<TSrc> e = ie.GetEnumerator();
            if (!e.MoveNext())
                return -1;

            int max_ix = 0;
            TArg v, max_val = fn(e.Current);
            if (!e.MoveNext())
                return max_ix;

            int i = 1;
            do
            {
                if ((v = fn(e.Current)).CompareTo(max_val) > 0)
                {
                    max_val = v;
                    max_ix = i;
                }
                i++;
            }
            while (e.MoveNext());
            return max_ix;
        }

        public static T Pop<T>(this List<T> input, int i = -1)
        {
            i = i == -1 ? input.Count - 1 : i;
            T t = input[i];
            input.RemoveAt(i);
            return t;
        }
    }
}