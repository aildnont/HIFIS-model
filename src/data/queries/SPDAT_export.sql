USE HIFIS
GO

SELECT  

	sit.IntakeID,
	cli.ClientID,
	ppl.CurrentAge,
	sitype.NameE as SPDAT_Type,
	ppt.NameE as PreScreenPeriod,
	sit.StartDateTime as SPDAT_Date,
	sit.LastUpdatedDate,
	orgs.OrganizationID,
	ServiceProvider = orgs.Name,
	Questions.AssessmentQuestionID,
	Questions.Component,
	REPLACE(Questions.QuestionE,',','') as QuestionE,	-- remove commas
	REPLACE(CAST(Questions.DescriptionE as VARCHAR(MAX)),',','') as DescriptionE, -- remove commas
 	REPLACE(
		COALESCE(	CAST(QA.ScoreValue as NVARCHAR),
					QA.ScoreYN,
					hht.NameE,
					cpt.NameE,
					hct.NameE,
					hhft.NameE
				)
			,',','') as ScoreValue,
	QA.RefusedYN,
	ss.TotalScore
FROM            
	HIFIS_SPDAT_Intake sit
	INNER JOIN HIFIS_SPDAT_ScoringSummary ss ON sit.IntakeID = ss.IntakeID 
	INNER JOIN HIFIS_Services serv ON serv.serviceID = sit.ServiceID
	INNER JOIN HIFIS_Client_Services cserv ON cserv.serviceID = serv.ServiceID
	INNER JOIN HIFIS_Clients cli ON cli.ClientID = cserv.ClientID 
	INNER JOIN HIFIS_People ppl ON ppl.personID = cli.personID
	INNER JOIN HIFIS_SPDAT_IntakeTypes sitype ON sitype.id = sit.IntakeType
	INNER JOIN HIFIS_ORganizations orgs ON serv.organizationID = orgs.organizationID
	LEFT OUTER JOIN	HIFIS_SPDAT_AssessmentPeriodTypes apt ON apt.ID = sit.AssessmentPeriodTypeID
	LEFT OUTER JOIN HIFIS_SPDAT_PreScreenPeriodTypes ppt ON ppt.id = sit.PreScreenPeriodTypeID
	INNER JOIN HIFIS_SPDAT_Intake_QuestionsAnswered QA ON qa.IntakeID = sit.IntakeID
	INNER JOIN HIFIS_SPDAT_AssessmentQuestions Questions ON QA.AssessmentQuestionID = Questions.AssessmentQuestionID
	LEFT OUTER JOIN HIFIS_SPDAT_HistoryofHousingTypes hht ON hht.ID = QA.DDHistoryofHousingTypeID
	LEFT OUTER JOIN HIFIS_SPDAT_CommonPlaceTypes cpt ON cpt.ID = QA.DDCommonPlaceTypeID
	LEFT OUTER JOIN HIFIS_SPDAT_HealthCareTypes hct ON hct.ID = QA.DDHealthCareTypeID
	LEFT OUTER JOIN HIFIS_SPDAT_HistoryofHousingFamilyTypes hhft ON hhft.ID = QA.DDHistoryofHousingFamilyTypeID
WHERE 
	ss.TOTALSCORE IS NOT NULL
AND 
	cli.ClientStateTypeID = 1 --Active Clients
AND 
 	orgs.ClusterID = 16 -- HIFIS Shared Cluster
AND
	sitype.NameE LIKE '%VI%' --VI-SPDATS ONLY

ORDER BY 
	cli.ClientID,sit.IntakeID,qa.AssessmentQuestionID
FOR JSON PATH, ROOT('VISPDATS')  --EXPORT IN JSON FORMAT 