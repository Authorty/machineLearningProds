using Microsoft.Data.DataView;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningProductPrediction
{

    public static class MachineLearning
    {

        public static string PredictNextProduct(List<ProgramProductData> productHistoricalData, ProgramProductData currentData)
        {
            // STEP 2: Create a ML.NET environment  
            MLContext mlContext = new MLContext();


            //var pipeline = mlContext.Transforms.Conversion.MapValueToKey("CurrentSaleDate")
            //    .Append(mlContext.Transforms.Conversion.MapValueToKey("LastSaleDate"))
            //    .Append(mlContext.Transforms.Conversion.MapValueToKey("SecondProduct"))
            //    .Append(mlContext.Transforms.Conversion.MapValueToKey("CurrentProduct"))
            //    .Append(mlContext.Transforms.Concatenate("Features", "Quantity", "Price"))
            //    .Append(mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(labelColumnName: "SecondProduct", featureColumnName: "Features"))




            //    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));


        var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
            .Append(mlContext.Transforms.Concatenate("Features", "Price", "LastSaleDate", "CurrentSaleDate", "Quantity", "companyName", "ProductsBoughtWith", "ProductType", "ProductCategory", "ProductGroup"))
            .AppendCacheCheckpoint(mlContext)
            .Append(mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(labelColumnName: "Label", featureColumnName: "Features"))
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var data = (IDataView)productHistoricalData;

            // STEP 4: Train your model based on the data set  
            var model = pipeline.Fit(data);

            var prediction = model.CreatePredictionEngine<ProgramProductData, PredictedLabel>(mlContext).Predict(
            currentData);

            return prediction.PredictedLabels;
        }
    }
}
