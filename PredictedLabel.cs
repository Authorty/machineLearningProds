using System;
using Microsoft.Data.DataView;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace MachineLearningProductPrediction
{
    class PredictedLabel
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabels;
    }
}
