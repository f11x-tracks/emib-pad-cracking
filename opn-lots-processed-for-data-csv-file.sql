SELECT 
          leh.lot AS ent_lot
         ,leh.operation AS ent_operation
         ,leh.entity AS entity
         ,leh.processed_wafer_count AS processed_wafer_count
         ,leh.lot_abort_flag AS lot_abort_flag
         ,leh.reticle AS reticle
         ,leh.lot_entity_process_duration AS lot_entity_process_duration
         ,leh.lot_process AS ent_lot_process
         ,Replace(Replace(Replace(Replace(Replace(Replace(p.product_description,',',';'),chr(9),' '),chr(10),' '),chr(13),' '),chr(34),''''),chr(7),' ') AS product_description
         ,lrc.product AS product
         ,lrc.dotprocess AS dotprocess
         ,lrc.route AS route
         ,lrc.lot_type AS lot_type
         ,lrc.oper_short_desc AS oper_short_desc
         ,l.hotlot AS current_hotlot
         ,lwr.recipe AS lot_recipe
         ,To_Char(leh.last_wafer_end_time,'yyyy-mm-dd hh24:mi:ss') AS last_wafer_end_date
         ,To_Char(leh.first_wafer_end_time,'yyyy-mm-dd hh24:mi:ss') AS first_wafer_end_date
         ,leh.lot_priority AS ent_lot_priority
FROM 
F_LotEntityHist leh
INNER JOIN F_Lot_Wafer_Recipe lwr ON lwr.recipe_id=leh.lot_recipe_id
INNER JOIN F_Lot_Run_card lrc ON lrc.lotoperkey = leh.lotoperkey
INNER JOIN F_Product p ON p.product=lrc.product AND p.facility = lrc.facility AND p.latest_version = 'Y'
INNER JOIN F_LOT l ON l.lot = lrc.lot
WHERE leh.last_wafer_end_time >= SYSDATE - 200
AND (leh.operation LIKE '194997' AND leh.entity LIKE 'SDJ%')
 OR (leh.operation LIKE '197573' AND leh.entity LIKE 'OXS%')
 OR (leh.operation LIKE '172748' AND leh.entity LIKE 'OXS%')
 OR (leh.operation LIKE '197573' AND leh.entity LIKE 'OXS%')
 OR (leh.operation LIKE '187614' AND leh.entity LIKE 'CVD%')
 OR (leh.operation LIKE '234511' AND leh.entity LIKE 'CVD%')
 