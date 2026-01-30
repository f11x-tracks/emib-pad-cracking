SELECT 
          leh.lot AS lot1
         ,wch.wafer AS wafer
         ,leh.entity AS lot_entity
         ,wch.slot AS slot
         ,wch.chamber AS chamber
         ,wch.state AS state
         ,To_Char(wch.start_time,'yyyy-mm-dd hh24:mi:ss') AS start_date
         ,To_Char(wch.end_time,'yyyy-mm-dd hh24:mi:ss') AS end_date
         ,lwr2.recipe AS lot_recipe
         ,leh.operation AS operation1
FROM 
F_LotEntityHist leh
INNER JOIN
F_WaferChamberHist wch
ON leh.runkey = wch.runkey
INNER JOIN F_Lot_Wafer_Recipe lwr2 ON lwr2.recipe_id=leh.lot_recipe_id
WHERE
              (wch.chamber LIKE '%CGCH%'
OR wch.chamber LIKE '%PC%')  
 AND      (leh.entity Like '%SDJ591%' OR leh.entity Like 'OXS%')
 AND (leh.OPERATION LIKE '194997' or leh.OPERATION LIKE '197573')
 AND      leh.run_txn_time >= TRUNC(SYSDATE) - 270 
 AND      (leh.lot LIKE 'W3179010%'
OR leh.lot LIKE 'W3519680%'
OR leh.lot LIKE 'W4139780%'
OR leh.lot LIKE 'W4299410%'
OR leh.lot LIKE 'W5052010%'
OR leh.lot LIKE 'W5064590%'
OR leh.lot LIKE 'W5064610%'
OR leh.lot LIKE 'W5070100%'
OR leh.lot LIKE 'W5070110%'
OR leh.lot LIKE 'W5070440%'
OR leh.lot LIKE 'W5070460%'
OR leh.lot LIKE 'W5083300%'
OR leh.lot LIKE 'W5083310%'
OR leh.lot LIKE 'W5083320%'
OR leh.lot LIKE 'W5083330%'
OR leh.lot LIKE 'W5083940%'
OR leh.lot LIKE 'W5083950%'
OR leh.lot LIKE 'W5083960%'
OR leh.lot LIKE 'W5084710%'
OR leh.lot LIKE 'W5095250%'
OR leh.lot LIKE 'W5101180%'
OR leh.lot LIKE 'W5101560%'
OR leh.lot LIKE 'W5101580%'
OR leh.lot LIKE 'W5101600%'
OR leh.lot LIKE 'W5113400%'
OR leh.lot LIKE 'W5113410%'
OR leh.lot LIKE 'W5114160%'
OR leh.lot LIKE 'W5125420%'
OR leh.lot LIKE 'W5125430%'
OR leh.lot LIKE 'W5125440%'
OR leh.lot LIKE 'W5125910%'
OR leh.lot LIKE 'W5125950%'
OR leh.lot LIKE 'W5125960%'
OR leh.lot LIKE 'W5136260%'
OR leh.lot LIKE 'W5136300%'
OR leh.lot LIKE 'W5136350%'
OR leh.lot LIKE 'W5136360%'
OR leh.lot LIKE 'W5141950%'
OR leh.lot LIKE 'W5141960%'
OR leh.lot LIKE 'W5141970%'
OR leh.lot LIKE 'W5142510%'
OR leh.lot LIKE 'W5149590%'
OR leh.lot LIKE 'W5149610%'
OR leh.lot LIKE 'W5155690%'
OR leh.lot LIKE 'W5155720%'
OR leh.lot LIKE 'W5156270%'
OR leh.lot LIKE 'W5156570%'
OR leh.lot LIKE 'W5163420%'
OR leh.lot LIKE 'W5163430%'
OR leh.lot LIKE 'W5163440%'
OR leh.lot LIKE 'W5170590%'
OR leh.lot LIKE 'W5170600%'
OR leh.lot LIKE 'W5170610%'
OR leh.lot LIKE 'W5182220%'
OR leh.lot LIKE 'W5191310%'
OR leh.lot LIKE 'W5195540%'
OR leh.lot LIKE 'W5203630%'
OR leh.lot LIKE 'W5203640%'
OR leh.lot LIKE 'W5204130%'
OR leh.lot LIKE 'W5204680%'
OR leh.lot LIKE 'W5210840%'
OR leh.lot LIKE 'W5210850%'
OR leh.lot LIKE 'W5210860%'
OR leh.lot LIKE 'W5210870%'
OR leh.lot LIKE 'W5211380%'
OR leh.lot LIKE 'W5211390%'
OR leh.lot LIKE 'W5216240%'
OR leh.lot LIKE 'W5216250%'
OR leh.lot LIKE 'W5223380%'
OR leh.lot LIKE 'W5223390%'
OR leh.lot LIKE 'W5234300%'
OR leh.lot LIKE 'W5240620%'
OR leh.lot LIKE 'W5240630%'
OR leh.lot LIKE 'W5240640%'
OR leh.lot LIKE 'W5240940%'
OR leh.lot LIKE 'W5240960%'
OR leh.lot LIKE 'W5241210%'
OR leh.lot LIKE 'W5241220%'
OR leh.lot LIKE 'W5241230%'
OR leh.lot LIKE 'W5251980%'
OR leh.lot LIKE 'W5252870%'
OR leh.lot LIKE 'W5253180%'
OR leh.lot LIKE 'W5253190%'
OR leh.lot LIKE 'W5253200%'
OR leh.lot LIKE 'W5253220%'
OR leh.lot LIKE 'W5263860%'
OR leh.lot LIKE 'W5264430%'
OR leh.lot LIKE 'W5264440%'
OR leh.lot LIKE 'W5264450%'
OR leh.lot LIKE 'W5265660%'
OR leh.lot LIKE 'W5266050%'
OR leh.lot LIKE 'W5266060%'
OR leh.lot LIKE 'W5266070%'
OR leh.lot LIKE 'W5266200%'
OR leh.lot LIKE 'W5270490%'
OR leh.lot LIKE 'W5270500%'
OR leh.lot LIKE 'W5270920%'
OR leh.lot LIKE 'W5271110%'
OR leh.lot LIKE 'W5282900%'
) 
 AND      (lwr2.recipe LIKE  '%') 