//------------------------------------------------------------------------------
// <auto-generated>
//     This code was generated from a template.
//
//     Manual changes to this file may cause unexpected behavior in your application.
//     Manual changes to this file will be overwritten if the code is regenerated.
// </auto-generated>
//------------------------------------------------------------------------------

namespace TestingProductPrediction
{
    using System;
    using System.Collections.Generic;
    
    public partial class csta_product_group_level_7_values
    {
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Usage", "CA2214:DoNotCallOverridableMethodsInConstructors")]
        public csta_product_group_level_7_values()
        {
            this.onca_product = new HashSet<onca_product>();
        }
    
        public string product_group_level_7_code { get; set; }
        public string description { get; set; }
        public string external_code { get; set; }
        public string active { get; set; }
        public string parent_group_level_code { get; set; }
        public string product_group_level_id { get; set; }
    
        public virtual csta_product_group_level_definition csta_product_group_level_definition { get; set; }
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Usage", "CA2227:CollectionPropertiesShouldBeReadOnly")]
        public virtual ICollection<onca_product> onca_product { get; set; }
    }
}
