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

        public static float ConvertStringToFloat(string value)
        {
            
            if (String.IsNullOrEmpty(value))
            {
                return 0.0f;
            }
            var valueBytes = System.Text.Encoding.ASCII.GetBytes(value);

            float floatValueReturn = System.BitConverter.ToSingle(valueBytes, 0);

            return floatValueReturn;

        }
        public static string PredictNextProduct(List<ProgramProductData> productHistoricalData, ProgramProductData currentData)
        {
            try
            {

            

            // STEP 2: Create a ML.NET environment  
            MLContext mlContext = new MLContext();



            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
            //.Append(mlContext.Transforms.Concatenate("Features", "Price", "LastSaleDate", "CurrentSaleDate", "Quantity", "companyName", "ProductsBoughtWith", "ProductType", "ProductCategory", "ProductGroup"))
            .Append(mlContext.Transforms.Concatenate("Features", "Price",  "CurrentSaleDate",  "companyName", "CompanyType"  , "ProductCategory", "ProductGroup"))
            .AppendCacheCheckpoint(mlContext)
            .Append(mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(labelColumnName: "Label", featureColumnName: "Features"))
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));


            var data = mlContext.Data.LoadFromEnumerable<ProgramProductData>(productHistoricalData);
            // STEP 4: Train your model based on the data set  
            var model = pipeline.Fit(data);

            var prediction = model.CreatePredictionEngine<ProgramProductData, PredictedLabel>(mlContext).Predict(
            currentData);

            return prediction.PredictedLabels;
            }
            catch (Exception e)
            {
                return e.InnerException.ToString();
                
            }
        }
    }
}
