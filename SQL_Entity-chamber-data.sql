SELECT 
          leh.lot AS lot1
         ,wch.waf3 AS waf3
         ,leh.entity AS lot_entity
         ,wch.slot AS slot
         ,wch.chamber AS chamber
         ,To_Char(leh.introduce_txn_time,'yyyy-mm-dd hh24:mi:ss') AS lot_intro_date
         ,wch.state AS state
         ,To_Char(wch.start_time,'yyyy-mm-dd hh24:mi:ss') AS start_date
         ,To_Char(wch.end_time,'yyyy-mm-dd hh24:mi:ss') AS end_date
         ,lwr2.recipe AS lot_recipe
         ,leh.lot_abort_flag AS lot_abort_flag
         ,leh.load_port AS lot_load_port
         ,leh.processed_wafer_count AS lot_processed_wafer_count
         ,leh.reticle AS lot_reticle
         ,lrc.rework AS rework
         ,leh.operation AS operation1
         ,leh.route AS lot_route
         ,leh.product AS lot_product
FROM 
F_LotEntityHist leh
INNER JOIN
F_WaferChamberHist wch
ON leh.runkey = wch.runkey
INNER JOIN F_Lot_Wafer_Recipe lwr2 ON lwr2.recipe_id=leh.lot_recipe_id
INNER JOIN F_Lot_Run_card lrc ON lrc.lotoperkey = wch.lotoperkey
WHERE
              (wch.chamber LIKE  '%')
 AND      (leh.entity LIKE 'OXS%' or leh.entity LIKE 'OXS%')
 AND      (leh.lot LIKE  'W%') 
 AND      lwr2.recipe Like '%' 
 AND      wch.start_time >= TRUNC(SYSDATE) - 1