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
    
    public partial class csta_product_group_level_definition
    {
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Usage", "CA2214:DoNotCallOverridableMethodsInConstructors")]
        public csta_product_group_level_definition()
        {
            this.csta_product_group_level_1_values = new HashSet<csta_product_group_level_1_values>();
            this.csta_product_group_level_2_values = new HashSet<csta_product_group_level_2_values>();
            this.csta_product_group_level_3_values = new HashSet<csta_product_group_level_3_values>();
            this.csta_product_group_level_4_values = new HashSet<csta_product_group_level_4_values>();
            this.csta_product_group_level_5_values = new HashSet<csta_product_group_level_5_values>();
            this.csta_product_group_level_6_values = new HashSet<csta_product_group_level_6_values>();
            this.csta_product_group_level_7_values = new HashSet<csta_product_group_level_7_values>();
            this.csta_product_group_level_8_values = new HashSet<csta_product_group_level_8_values>();
        }
    
        public string product_group_level_id { get; set; }
        public string description { get; set; }
        public Nullable<int> sort_order { get; set; }
        public string active { get; set; }
    
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Usage", "CA2227:CollectionPropertiesShouldBeReadOnly")]
        public virtual ICollection<csta_product_group_level_1_values> csta_product_group_level_1_values { get; set; }
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Usage", "CA2227:CollectionPropertiesShouldBeReadOnly")]
        public virtual ICollection<csta_product_group_level_2_values> csta_product_group_level_2_values { get; set; }
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Usage", "CA2227:CollectionPropertiesShouldBeReadOnly")]
        public virtual ICollection<csta_product_group_level_3_values> csta_product_group_level_3_values { get; set; }
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Usage", "CA2227:CollectionPropertiesShouldBeReadOnly")]
        public virtual ICollection<csta_product_group_level_4_values> csta_product_group_level_4_values { get; set; }
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Usage", "CA2227:CollectionPropertiesShouldBeReadOnly")]
        public virtual ICollection<csta_product_group_level_5_values> csta_product_group_level_5_values { get; set; }
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Usage", "CA2227:CollectionPropertiesShouldBeReadOnly")]
        public virtual ICollection<csta_product_group_level_6_values> csta_product_group_level_6_values { get; set; }
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Usage", "CA2227:CollectionPropertiesShouldBeReadOnly")]
        public virtual ICollection<csta_product_group_level_7_values> csta_product_group_level_7_values { get; set; }
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Usage", "CA2227:CollectionPropertiesShouldBeReadOnly")]
        public virtual ICollection<csta_product_group_level_8_values> csta_product_group_level_8_values { get; set; }
        public virtual csta_product_group_level_definition_table csta_product_group_level_definition_table { get; set; }
    }
}