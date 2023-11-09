--CREATE EXTENSION btree_gist;

--select n.* from data_n1 n where aircraft ='VQ-BDQ';

select a.reportts, dn 
	FROM a320_a321_neo_full_acms_parameters a 
	CROSS JOIN LATERAL (
		SELECT n1.recorded_dt 
		from data_n1 n1
		WHERE n1.aircraft = 'VQ-BDQ'
		order by n1.recorded_dt <-> a.reportts 
		limit 1
	) AS dn
	WHERE a.acnum = 'VQ-BDQ' -- and dn.aircraft = 'VQ-BDQ'
	;