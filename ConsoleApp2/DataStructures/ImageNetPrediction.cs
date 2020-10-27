using Microsoft.ML.Data;

namespace ConsoleApp2.DataStructures
{
    public class ImageNetPrediction
    {
        [ColumnName("output")]
        public float[] PredictedLabels;
    }
}