using System;
using System.IO;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using Microsoft.ML;
using ConsoleApp2.DataStructures;
using Microsoft.ML.Data;
using Emgu.CV;
using Emgu.CV.Structure;

namespace ConsoleApp2
{
	class Program
	{
        static double CosineDistance(float[] a, float[] b)
        {
            double num = 0;
            double denom1 = 0, denom2 = 0;
            foreach (var el in a)
            {
                denom1 += el * el;
            }
            foreach (var el in b)
            {
                denom2 += el * el;
            }
            for (int i = 0; i < a.Length; ++i)
            {
                num += a[i] * b[i];
            }
            return 1 - num / Math.Sqrt(denom1 * denom2);
        }

        static double EuclideanDistance(float[] a, float[] b)
        {
            double num = 0;
            for (int i = 0; i < a.Length; ++i)
            {
                num += (a[i] - b[i]) * (a[i] - b[i]);
            }
            return Math.Sqrt(num);
        }

        static void Main(string[] args)
        {
            var modelFilePath = "..\\..\\..\\assets\\onnx_reid_model_best.onnx";
            var imagesFolder = "..\\..\\..\\assets\\images";

            MLContext mlContext = new MLContext();

            try
            {
                IEnumerable<ImageNetData> images = ImageNetData.ReadFromFile(imagesFolder);
                IDataView imageDataView = mlContext.Data.LoadFromEnumerable(images);
                if (!Directory.Exists(Path.Join(imagesFolder, "captureImages"))) {
                    Directory.CreateDirectory(Path.Join(imagesFolder, "captureImages"));
                }
                var modelScorer = new OnnxModelScorer(imagesFolder, modelFilePath, mlContext);
                List<float[]> initialFeatures = modelScorer.Score(imageDataView).ToList();

                VideoCapture capture = new VideoCapture();
                double bestScore1 = 1, bestScore2 = 1;
                while (!(Console.KeyAvailable))
                {
                    var frame = capture.QueryFrame();
                    if (frame != null)
                    {
                        var image = frame.ToImage<Bgr, byte>();

                        image.Save(Path.Join(imagesFolder, "captureImages", "capture.jpg"));

                        images = ImageNetData.ReadFromFile(Path.Join(imagesFolder, "captureImages"));
                        imageDataView = mlContext.Data.LoadFromEnumerable(images);
                        List<float[]> features = modelScorer.Score(imageDataView).ToList();
                        var score1 = CosineDistance(initialFeatures[11], features[0]);
                        var score2 = CosineDistance(initialFeatures[12], features[0]);
                        bestScore1 = Math.Min(score1, bestScore1);
                        bestScore2 = Math.Min(score2, bestScore2);

                        Console.WriteLine($"{score1} {score2}");
                        image.Dispose();
                        frame.Dispose();
                        File.Delete(Path.Join(imagesFolder, "captureImages", "capture.jpg"));
                    }
                }
                Console.WriteLine($"Best score1 = {bestScore1}");
                Console.WriteLine($"Best score2 = {bestScore2}");

                //for (int i = 0; i < 10; ++i)
                //{
                //    Console.Write(features[0][i].ToString() + ' ');
                //}
                //Console.WriteLine();
                //for (int i = 0; i < 10; ++i)
                //{
                //    Console.Write(features[1][i].ToString() + ' ');
                //}
                //Console.WriteLine();

                //for (int i = 11; i < 17; ++i)
                //{
                //    for (int j = 11; j < 17; ++j)
                //    {
                //        Console.Write("{0:0.00} ", CosineDistance(features[i], features[j]));
                //    }
                //    Console.WriteLine();
                //}

            }
            catch (Exception e)
            {
                Console.WriteLine(e.Message);
            }
        }
    }
}
