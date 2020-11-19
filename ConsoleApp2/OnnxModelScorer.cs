using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using ConsoleApp2.DataStructures;
using System.Drawing;
using System.IO;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Image;
using static Microsoft.ML.Transforms.Image.ImagePixelExtractingEstimator;
using Microsoft.ML.Transforms.Onnx;

namespace ConsoleApp2
{
    class OnnxModelScorer
    {
        private readonly string imagesFolder;
        private readonly string modelLocation;
        private readonly MLContext mlContext;

        public OnnxModelScorer(string imagesFolder, string modelLocation, MLContext mlContext)
        {
            this.imagesFolder = imagesFolder;
            this.modelLocation = modelLocation;
            this.mlContext = mlContext;
        }

        public struct ImageNetSettings
        {
            public const int imageHeight = 224;
            public const int imageWidth = 112;
        }

        public struct ReidModelSettings
        {
            // for checking Tiny yolo2 Model input and  output  parameter names,
            //you can use tools like Netron, 
            // which is installed by Visual Studio AI Tools

            // input tensor name
            public const string ModelInput = "input";

            // output tensor name
            public const string ModelOutput = "output";
        }

        private ITransformer LoadModel(string modelLocation)
        {
            Console.WriteLine("Read model");
            Console.WriteLine($"Model location: {modelLocation}");
            Console.WriteLine($"Default parameters: image size=({ImageNetSettings.imageWidth}, {ImageNetSettings.imageHeight})");

            // Create IDataView from empty list to obtain input data schema
            var data = mlContext.Data.LoadFromEnumerable(new List<ImageNetData>());
            //var imagesDataFile = "C:\\Users\\Asus\\.jupyter\\images\\images.tsv";
            //var data = mlContext.Data.CreateTextLoader(new TextLoader.Options()
            //{
            //    Columns = new[]
            //        {
            //            new TextLoader.Column("ImagePath", DataKind.String, 0),
            //            new TextLoader.Column("Name", DataKind.String, 1),
            //        }
            //}).Load(imagesDataFile);

            // var imagesFolder = Path.GetDirectoryName(imagesDataFile);
            // Define scoring pipeline
            var pipeline = mlContext.Transforms.LoadImages(outputColumnName: "ImageObject", imageFolder: "", inputColumnName: nameof(ImageNetData.ImagePath))
                           .Append(mlContext.Transforms.ResizeImages(outputColumnName: "ImageResized", imageWidth: ImageNetSettings.imageWidth, imageHeight: ImageNetSettings.imageHeight, inputColumnName: "ImageObject"))
                           .Append(mlContext.Transforms.ExtractPixels("Red", "ImageResized", colorsToExtract: ColorBits.Red, offsetImage: 0.485f * 255, scaleImage: 1 / (0.299f * 255)))
                           .Append(mlContext.Transforms.ExtractPixels("Green", "ImageResized", colorsToExtract: ColorBits.Green, offsetImage: 0.456f * 255, scaleImage: 1 / (0.224f * 255)))
                           .Append(mlContext.Transforms.ExtractPixels("Blue", "ImageResized", colorsToExtract: ColorBits.Blue, offsetImage: 0.406f * 255, scaleImage: 1 / (0.225f * 255)))
                           .Append(mlContext.Transforms.Concatenate(ReidModelSettings.ModelInput, "Red", "Green", "Blue"))
                           // .Append(new OnnxScoringEstimator(mlContext, new string[] { @"mobile"}, new string[] { "data" }, modelLocation))
                           //.Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: "Pixels", inputColumnName: "Pixels"))
                           .Append(mlContext.Transforms.ApplyOnnxModel(modelFile: modelLocation, outputColumnNames: new[] { ReidModelSettings.ModelOutput }, inputColumnNames: new[] { ReidModelSettings.ModelInput }));
                           //.Append(mlContext.Transforms.DnnFeaturizeImage(ReidModelSettings.ModelOutput, m => m.ModelSelector.ResNet50(mlContext, m.OutputColumn, m.InputColumn), "Pixels"));

            //var pipeline1 = mlContext.Transforms.LoadImages(outputColumnName: "ImageObject", imageFolder: imagesFolder, inputColumnName: "ImagePath");
            //data = pipeline1.Fit(data).Transform(data);

            //var pipeline2 = mlContext.Transforms.ResizeImages(outputColumnName: "ImageObject", imageWidth: ImageNetSettings.imageWidth, imageHeight: ImageNetSettings.imageHeight, inputColumnName: "ImageObject", ImageResizingEstimator.ResizingKind.IsoCrop);
            //PrintColumns(pipeline2.Fit(data).Transform(data));
            // Fit scoring pipeline
            var model = pipeline.Fit(data);

            return model;
        }

        private IEnumerable<float[]> PredictDataUsingModel(IDataView testData, ITransformer model)
        {
            Console.WriteLine($"Images location: {imagesFolder}");
            Console.WriteLine("");
            Console.WriteLine("=====Identify the objects in the images=====");
            Console.WriteLine("");

            IDataView scoredData = model.Transform(testData);


            IEnumerable<float[]> features = scoredData.GetColumn<float[]>(ReidModelSettings.ModelOutput);

            return features;
        }

        public IEnumerable<float[]> Score(IDataView data)
        {
            var model = LoadModel(modelLocation);

            return PredictDataUsingModel(data, model);
        }

        private static void PrintColumns(IDataView transformedData)
        {
            // The transformedData IDataView contains the loaded images now.
            Console.WriteLine("{0, -25} {1, -25} {2, -25}", "ImagePath", "Name",
                "ImageObject");

            using (var cursor = transformedData.GetRowCursor(transformedData
                .Schema))
            {
                // Note that it is best to get the getters and values *before*
                // iteration, so as to facilitate buffer sharing (if applicable),
                // and column-type validation once, rather than many times.
                ReadOnlyMemory<char> imagePath = default;
                ReadOnlyMemory<char> name = default;
                Bitmap imageObject = null;

                var imagePathGetter = cursor.GetGetter<ReadOnlyMemory<char>>(cursor
                    .Schema["ImagePath"]);

                var nameGetter = cursor.GetGetter<ReadOnlyMemory<char>>(cursor
                    .Schema["Name"]);

                var imageObjectGetter = cursor.GetGetter<Bitmap>(cursor.Schema[
                    "ImageObject"]);

                while (cursor.MoveNext())
                {

                    imagePathGetter(ref imagePath);
                    nameGetter(ref name);
                    imageObjectGetter(ref imageObject);

                    Console.WriteLine("{0, -25} {1, -25} {2, -25}", imagePath, name,
                        imageObject.PhysicalDimension);
                    Console.WriteLine(imageObject.GetPixel(0, 0));
                }

                // Dispose the image.
                imageObject.Dispose();
            }
        }
    }
}
