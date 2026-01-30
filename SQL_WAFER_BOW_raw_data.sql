SELECT 
          a0.lot AS lot
         ,a0.operation AS spc_operation
         ,To_Char(a0.data_collection_time,'yyyy-mm-dd hh24:mi:ss') AS lot_data_collect_date
         ,a5.value AS chart_value
         ,a5.wafer AS chart_wafer
         ,a5.incontrol_flag AS incontrol_flag
         ,a5.status AS chart_pt_status
         ,a5.chart_type AS chart_type
         ,a5.spc_chart_subset AS spc_chart_subset
         ,a5.test_name AS chart_test_name
FROM 
P_SPC_LOT a0
INNER JOIN P_SPC_SESSION a2 ON a2.spcs_id = a0.spcs_id AND a2.data_collection_time = a0.data_collection_time
INNER JOIN P_SPC_MEASUREMENT_SET a3 ON a3.spcs_id = a2.spcs_id
INNER JOIN P_SPC_CHART_POINT a5 ON a5.spcs_id = a3.spcs_id AND a5.measurement_set_name = a3.measurement_set_name
WHERE
              a0.operation = '187622' 
 AND      a0.data_collection_time >= TRUNC(SYSDATE) - 200
 AND a5.test_name like '257%'