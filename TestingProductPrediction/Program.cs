using MachineLearningProductPrediction;
using System;
using System.Collections.Generic;
using System.Linq;

namespace TestingProductPrediction
{
    class Program
    {
        static void Main(string[] args)
        {

            onc_pg_famEntities db = new onc_pg_famEntities();
            var data = db.cstd_company_sales_history.Take(15000).ToList();


            List<ProgramProductData> pList = new List<ProgramProductData>();

            foreach (var item in data)
            {



                var companyName = item.oncd_company.company_name_1;
                var companyNameBytes = System.Text.Encoding.ASCII.GetBytes(companyName);

                pList.Add(new ProgramProductData
                {
                    companyName = MachineLearning.ConvertStringToFloat(item.oncd_company.company_name_1),
                    CompanyType = MachineLearning.ConvertStringToFloat(item.oncd_company.company_type_code),
                    ProductCategory = MachineLearning.ConvertStringToFloat(item.onca_product.cst_product_category_code),
                    ProductGroup = MachineLearning.ConvertStringToFloat(item.onca_product.cst_product_group_code),
                    CurrentSaleDate = MachineLearning.ConvertStringToFloat(item.sales_date.ToString()),
                    Price = (float)item.sales_revenue,
                    Label = item.onca_product.cst_alternate_description,
                    //ProductType = MachineLearning.ConvertStringToFloat(item.onca_product.typ),
                    
                });

            }


            var prediction = MachineLearning.PredictNextProduct(pList, new ProgramProductData {
                companyName = MachineLearning.ConvertStringToFloat("Navarro Dist Center                                                                                 "),
                CompanyType = MachineLearning.ConvertStringToFloat("54C2EDA044"),
                ProductCategory = MachineLearning.ConvertStringToFloat(""),
                ProductGroup = MachineLearning.ConvertStringToFloat(""),
                CurrentSaleDate = MachineLearning.ConvertStringToFloat(DateTime.Now.ToString()),
                Price = (float)23.24,
                Label = "CG TRUblnd LMU Per Bg U2A                         ",
            });

            Console.WriteLine(prediction);
            var a = "";
        }
    }
}
