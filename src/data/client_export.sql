USE [DATABASE] 
GO

;WITH Consent AS(

SELECT * FROM
 (
 SELECT 
			ClientID, 
			Con.ConsentID, 
			ConsentTypeID, 
			ORgID,
			ServiceProvider = HIFIS_Organizations.Name,
			ConsentType = HIFIS_ConsentTypes.NameE, 
			ExpiryDate = min(ExpiryDate),
			Row# = ROW_NUMBER() OVER(PARTITION BY ClientID order by Case when ConsentTypeID != 2 then 1 else 2 end)
        FROM 
			HIFIS_Consent Con
			INNER JOIN HIFIS_Consent_ServiceProviders ON Con.ConsentID = HIFIS_Consent_ServiceProviders.ConsentID
			INNER JOIN HIFIS_ConsentTypes ON Con.ConsentTypeID = HIFIS_ConsentTypes.ID
			INNER JOIN HIFIS_Organizations ON HIFIS_Consent_ServiceProviders.OrgID = HIFIS_Organizations.OrganizationID
        WHERE 
			(GETDATE() BETWEEN Con.StartDate AND Con.ExpiryDate OR (GETDATE() >= Con.StartDate AND Con.ExpiryDate IS NULL))
        GROUP BY 
			ClientID,ConsentTypeID, Con.ConsentID,Orgid,NameE,HIFIS_Organizations.Name
		) as abc
    WHERE 
		Row# = 1

),
HOUSINGPLACEMENTS AS (

	SELECT 
		HousePlacementID,
		PrimaryClientID,
		MovedInDate
	--INTO
	--	HousingPlacements
	FROM
	(
		SELECT 
			ROW = ROW_NUMBER() OVER (PARTITION BY PrimaryClientID ORDER BY DateSearchStarted DESC, MovedInDate Desc),
			HousePlacementID,
			PrimaryClientID,
			DateSearchStarted,
			DateHousingSecured,
			DateExpectedMoveIn,
			MovedInDate,
			DateSearchEnded,
			FollowUpCompleteYN
		FROM
			HIFIS_HousePlacements a
		WHERE
			SERVICEID NOT IN (SELECT serviceID FROM HIFIS_Services WHERE OrganizationID IN (SELECT OrganizationID  FROM HIFIS_Organizations WHERE ClusterID <> 16))
	) t
	WHERE Row = 1
	AND MovedInDate IS NOT NULL
)
,
Families AS
(

	SELECT  
		HIFIS_People_Groups.PeopleGroupID,
		HIFIS_People_Groups.[GroupID], 
		HIFIS_People_Groups.[DateEnd], 
		HIFIS_People_Groups.[DateStart], 
		HIFIS_People_Groups.[GroupRoleTypeID], 
		HIFIS_People_Groups.[GroupHeadYN], 
		HIFIS_People_Groups.[ServiceFee], 
		HIFIS_People_Groups.[EmergencyContactYN], 
		HIFIS_People_Groups.[PeopleRelationshipTypeID], 
		HIFIS_People_Groups.[HifisRowVersion], 
		HIFIS_People_Groups.[PersonID], 
		HIFIS_People_Groups.[CreatedDate], 
		HIFIS_People_Groups.[LastUpdatedDate], 
		HIFIS_People_Groups.[LastUpdatedBy], 
		HIFIS_People_Groups.[CreatedBy],
		HIFIS_PeopleRelationshipTypes.NameE as RelationshipType
	
	FROM
		HIFIS_People_Groups
		INNER JOIN HIFIS_Groups ON HIFIS_People_Groups.GroupID = HIFIS_GROUPS.GroupID
		INNER JOIN HIFIS_PeopleRelationshipTypes ON HIFIS_People_Groups.PeopleRelationshipTypeID = HIFIS_PeopleRelationshipTypes.id
	WHERE 
	( EXISTS 
			(	-- role type 11 = clients
				SELECT 
				1 AS [C1]
			
				FROM HIFIS_People_PeopleRoles 
				WHERE ([PersonID] = HIFIS_People_PeopleRoles.PersonID) 
				AND (11 = HIFIS_People_PeopleRoles.PeopleRoleTypeID))
				) 
	
		AND 
	(
		(HIFIS_People_Groups.DateEnd IS NULL) 
		OR 
		(HIFIS_People_Groups.DateEnd > GetDAte())
	) 
	AND 
	(
		(HIFIS_Groups.DateEnd IS NULL) 
		OR 
		(HIFIS_Groups.DateEnd > GetDate())
	) 
	AND HIFIS_People_Groups.GroupRoleTypeID <> 9
	

),
Incomes
AS
(
		SELECT * FROM
		(
			SELECT
				--ROW_NUMBER() OVER (PARTITION BY HIFIS_ClientINComes.clientID,HIFIS_ClientIncomes.IncomeTypeID ORDER BY DateStart DESC) as RowNumber,
				HIFIS_ClientIncomes.ClientID,
				IncomeTypeID,
				ClientIncomeID,
				HIFIS_IncomeTypes.NameE,
				MonthlyAmount,
				DateStart,
				DateEnd
			FROM 
				HIFIS_ClientIncomes
				INNER JOIN HIFIS_IncomeTypes ON HIFIS_ClientIncomes.IncomeTypeID = HIFIS_IncomeTypes.ID
			WHERE 
				HIFIS_ClientIncomes.createdDate IS NULL
			--AND
			--	ClientIncomeID NOT IN (Select ClientIncomeID FROM CTE)

		) t

),

Expenses AS
(
	SELECT 
		HIFIS_ClientExpenses.ClientExpenseID,
		HIFIS_ClientExpenses.ClientID,
		HIFIS_ClientExpenses.ExpenseTypeID,
		ExpenseType = HIFIS_ExpenseTypes.NameE,
		HIFIS_ClientExpenses.DateStart,
		HIFIS_ClientExpenses.DateEnd,
		HIFIS_ClientExpenses.PayFrequencyTypeID,
		ExpenseFrequency = HIFIS_PayFrequencyTypes.NameE,
		HIFIS_ClientExpenses.ExpenseAmount,
		HIFIS_ClientExpenses.IsEssentialYN
	 FROM 
		HIFIS_ClientExpenses
		INNER JOIN HIFIS_ExpenseTypes ON HIFIS_ClientExpenses.ExpenseTypeID = HIFIS_ExpenseTypes.ID
		INNER JOIN HIFIS_PayFrequencyTypes ON HIFIS_ClientExpenses.PayFrequencyTypeID = HIFIS_PayFrequencyTypes.ID

),
HealthIssues AS
(
		SELECT 
			HIFIS_HealthIssues.HealthIssueID,
			HIFIS_HealthIssues.ClientID,
			HIFIS_HealthIssues.HealthIssueTypeID,
			HealthIssue = HIFIS_HealthIssueTypes.NameE,
			DateFrom,
			DateTo,
			Description,
			Symptoms,
			Medication,
			Treatment,
			SelfReportedYN,
			SuspectedYN,
			DiagnosedYN,
			ContagiousYN,
			HIFIS_Medications.MedicationName
		FROM
			HIFIS_HealthIssues
			INNER JOIN HIFIS_HealthIssueTypes ON HIFIS_HealthIssues.HealthIssueTypeID = HIFIS_HealthIssueTypes.ID
			LEFT OUTER JOIN HIFIS_Medications ON HIFIS_HealthIssues.HealthIssueID = HIFIS_Medications.HealthIssueID

),

AssestsandLiabilities as
(
		SELECT
			HIFIS_LiabilitiesOrAssests.LiabilityOrAssetID,
			HIFIS_LiabilitiesOrAssests.ClientID,
			Type='Asset',
			HIFIS_LiabilitiesOrAssests.DateStart,
			HIFIS_LiabilitiesOrAssests.DateEnd,
			HIFIS_LiabilitiesOrAssests.Description,
			HIFIS_LiabilitiesOrAssests.Amount,
			HIFIS_AssetTypes.NameE

	FROM 
			HIFIS_LiabilitiesOrAssests
			LEFT OUTER JOIN HIFIS_AssetTypes ON HIFIS_LiabilitiesOrAssests.AssetTypeID = HIFIS_AssetTypes.ID
	WHERE AssetTypeID IS NOT NULL

UNION

		SELECT
			HIFIS_LiabilitiesOrAssests.LiabilityOrAssetID,
			HIFIS_LiabilitiesOrAssests.ClientID,
			Type='Liability',
			HIFIS_LiabilitiesOrAssests.DateStart,
			HIFIS_LiabilitiesOrAssests.DateEnd,
			HIFIS_LiabilitiesOrAssests.Description,
			HIFIS_LiabilitiesOrAssests.Amount,
			HIFIS_LiabilityTypes.NameE
		FROM 
			HIFIS_LiabilitiesOrAssests
			LEFT OUTER JOIN HIFIS_LiabilityTypes ON HIFIS_LiabilitiesOrAssests.AssetTypeID = HIFIS_LiabilityTypes.ID
		WHERE 
			LiabilityTypeID IS NOT NULL

),
WatchConcerns AS
(
		SELECT * FROM 
		(
			SELECT
				Row# = ROW_NUMBER() OVER (PARTITION BY ClientID ORDER BY DateStart DESC) ,
				ClientWatchConcernID,
				ClientID,
				HIFIS_Client_WatchConcerns.WatchConcernTypeID,
				WatchConcern = HIFIS_WatchConcernTypes.NameE,
				HIFIS_Client_WatchConcerns.DateStart,
				HIFIS_Client_WatchConcerns.DateEnd,
				HIFIS_Client_WatchConcerns.Comments
			FROM 
				HIFIS_Client_WatchConcerns 
				INNER JOIN HIFIS_WatchConcernTypes ON HIFIS_Client_WatchConcerns.WatchConcernTypeID = HIFIS_WatchConcernTypes.ID
			WHERE ID = 1000
) t
WHERE ROW#=1
	
),
ContributingFactors AS
(
		SELECT 
			ClientContributingFactorID,
			ClientID,
			ContributingFactor = HIFIS_ContributingFactorTypes.NameE,
			DateStart,
			DateEnd
		FROM 
			HIFIS_Client_ContributingFactor
			INNER JOIN HIFIS_ContributingFactorTypes ON HIFIS_Client_ContributingFactor.ContributingTypeID = HIFIS_ContributingFactorTypes.ID
),
BehavioralRiskFactors AS
(

		SELECT 
			ClientBehaviouralFactorID,
			ClientID,
			BehavioralFactor = HIFIS_BehaviouralFactorTypes.NameE,
			Severity = HIFIS_ProbabilityTypes.NameE,
			DateStart,
			DateEnd
		FROM 
			HIFIS_Client_BehaviouralFactor
			INNER JOIN HIFIS_BehaviouralFactorTypes ON HIFIS_Client_BehaviouralFactor.BehavioralTypeID = HIFIS_BehaviouralFactorTypes.ID
			INNER JOIN HIFIS_ProbabilityTypes ON HIFIS_Client_BehaviouralFactor.ProbabilityTypeID = HIFIS_ProbabilityTypes.ID

),
LifeEvents AS
(
		SELECT 
			PeopleLifeEventID,
			PersonID,
			HIFIS_LifeEventsTypes.ID,
			LifeEvent = HIFIS_LifeEventsTypes.NameE,
			DateStart,
			DateEnd
		FROM
			HIFIS_People_LifeEvents
			INNER JOIN HIFIS_LifeEventsTypes ON HIFIS_People_LifeEvents.LifeEventTypeID = HIFIS_LifeEventsTypes.ID
		WHERE LifeEventTypeID <> 1005
	
)
,VISPDATS AS
	

(	
	SELECT  
			 HIFIS_Clients.ClientID,
			 HIFIS_People.CurrentAge,
			 SPDAT_Type = HIFIS_SPDAT_IntakeTypes.NameE,
			 PreScreenPeriod = HIFIS_SPDAT_PreScreenPeriodTypes.NameE,
			 AssesmentPeriod = HIFIS_SPDAT_AssessmentPeriodTypes.NameE,
			 
			 SPDAT_Date = HIFIS_SPDAT_INtake.StartDateTime,
			 HIFIS_SPDAT_Intake.LastUpdatedDate,
			 HIFIS_Organizations.OrganizationID,
			 HIFIS_Organizations.Name as ServiceProvider,
			 --Questions.AssessmentQuestionID,
			 --Questions.QuestionE,
			 --Questions.DescriptionE,
			 --QA.ScoreValue,
			 HIFIS_SPDAT_ScoringSummary.TotalScore
			--RowNumber = ROW_NUMBER() OVER(PARTITION BY HIFIS_Clients.clientID ORDER BY HIFIS_Clients.ClientID,HIFIS_SPDAT_Intake.LastUpdatedDate DESC)
		FROM            
			HIFIS_SPDAT_Intake 
			INNER JOIN HIFIS_SPDAT_ScoringSummary ON HIFIS_SPDAT_Intake.IntakeID = HIFIS_SPDAT_ScoringSummary.IntakeID 
			INNER JOIN HIFIS_Services ON HIFIS_SPDAT_Intake.ServiceID = HIFIS_Services.ServiceID 
			INNER JOIN HIFIS_Client_Services ON HIFIS_Services.ServiceID = HIFIS_Client_Services.ServiceID
			INNER JOIN HIFIS_Clients ON HIFIS_Client_Services.ClientID = HIFIS_Clients.ClientID 
			INNER JOIN HIFIS_People ON HIFIS_CLients.PersonID = HIFIS_People.PersonID 
			INNER JOIN HIFIS_SPDAT_IntakeTypes ON HIFIS_SPDAT_Intake.IntakeType = HIFIS_SPDAT_IntakeTypes.ID 
			INNER JOIN HIFIS_ORganizations ON HIFIS_Services.OrganizationID = HIFIS_Organizations.OrganizationID 
			LEFT OUTER JOIN	HIFIS_SPDAT_AssessmentPeriodTypes ON HIFIS_SPDAT_Intake.AssessmentPeriodTypeID = HIFIS_SPDAT_AssessmentPeriodTypes.ID 
			LEFT OUTER JOIN HIFIS_SPDAT_PreScreenPeriodTypes ON HIFIS_SPDAT_Intake.PreScreenPeriodTypeID = HIFIS_SPDAT_PreScreenPeriodTypes.ID
			--INNER JOIN HIFIS_SPDAT_Intake_QuestionsAnswered QA ON HIFIS_SPDAT_Intake.IntakeID = QA.IntakeID
			--INNER JOIN HIFIS_SPDAT_AssessmentQuestions Questions ON QA.AssessmentQuestionID = Questions.AssessmentQuestionID
		WHERE 
			TOTALSCORE IS NOT NULL
		--	TotalScore > 8
		AND 
			HIFIS_CLients.ClientStateTypeID = 1
		AND 
			HIFIS_Organizations.OrganizationID IN (SELECT OrganizationID FROM HIFIS_Organizations WHERE ClusterID = 16)
		AND
			HIFIS_SPDAT_IntakeTypes.NameE LIKE '%VI%'



)

,Medications AS
(
		SELECT 
			medicationID,
			ClientID,
			MedicationName,
			Dosage,
			DateStart,
			DateEnd
		FROM
			HIFIS_Medications
		WHERE HealthIssueID IS NULL


),
Diets AS
(
			SELECT 
				HIFIS_ClientDiets.ClientDietID,
				HIFIS_ClientDiets.ClientID,
				DietCatetory = HIFIS_DietCategoryTypes.NameE,
				FoodType = HIFIS_DietFoodItemTypes.NameE,
				HIFIS_ClientDiets.AvoidedDietYN

		 FROM
				HIFIS_ClientDiets
				INNER JOIN HIFIS_DietCategoryTypes ON HIFIS_ClientDiets.DietCategoryTypeID = HIFIS_DietCategoryTypes.ID
				INNER JOIN HIFIS_DietFoodItemTypes ON HIFIS_ClientDiets.DietFoodItemTypeID = HIFIS_DietFoodItemTypes.ID

)




SELECT 

		vw_ClientsServices.ServiceID,
		vw_ClientsServices.ClientID,
		Consent.ConsentType,
		FamilyID = Families.GroupID,
		--Families.PeopleRelationshipTypeID,
		Families.RelationshipType,
		HIFIS_People.DOB,
		vw_ClientsServices.CurrentAge,
		--CurrentAge = DATEDIFF(Day,HIFIS_People.DOB,GetDate()) / 365.2422,
		Gender = Gender_En,
		AboriginalIndicator = AboriginalIndicator_En,
		Citizenship = Citizenship_En,
		VeteranStatus = VeteranStatus_En,
		CountryOfBirth = HIFIS_CountryTypes.NameE,
		ProvinceOfBirth = COALESCE(HIFIS_ProvinceTypes.NameE,ProvinceFreeText),
		CityOfBirth = COALESCE(HIFIS_CityTypes.NameE,CityFreeText),
		GeoRegion = HIFIS_GeoRegionTypes.NameE,
		HairColour = HIFIS_HairColorTypes.NameE,
		EyeColour = HIFIS_EyeColorTypes.NameE,
		ClientHeightCM = ClientHeight,
		ClientWeightKG = ClientWeight,
		ServiceType = ServiceType_En,
		vw_ClientsServices.OrganizationID,
		vw_ClientsServices.OrganizationName,
		ServiceStartDate = vw_ClientsServices.DateStart,
		ServiceEndDate = vw_Clientsservices.DateEnd,
		ReasonForService = ReasonForService_En,
		CurrentlyHoused = CASE WHEN HousingPlacements.PrimaryClientID IS NOT NULL THEN 'Yes' Else 'No' END,
		MovedInDate,
		IncomeType = Incomes.NameE,
		Incomes.MonthlyAmount,
		IncomeStartDate = Incomes.DateStart,
		IncomeEndDate = Incomes.DateEnd,
		ExpenseType = Replace(Expenses.ExpenseType,',',''),
		ExpenseStartDate = Expenses.DateStart,
		ExpenseEndDate = Expenses.DateEnd,
		Expenses.Expensefrequency,
		Expenses.ExpenseAmount,
		Expenses.IsEssentialYN,
		EducationLevel = HIFIS_EducationLevelTypes.NameE,
		EducationStartDate = HIFIS_ClientEducationLevels.DateStart,
		EducationEndDate = HIFIS_ClientEducationLevels.DateEnd,
		HealthIssues.HealthIssue,
		HealthIssues.DiagnosedYN,
		HealthIssues.SelfReportedYN,
		HealthIssues.SuspectedYN,
		HealthIssueStart = HealthIssues.DateFrom,
		HealthIssueEnd = HealthIssues.DateTo,
		HealthIssueMedicationName = HealthIssues.MedicationName,
		OtherMedications = Medications.MedicationName,
		WatchConcerns.WatchConcern,
		ContributingFactors.ContributingFactor,
		ContributingFactorDateStart = ContributingFactors.DateStart,
		ContributingFactorDateEnd = ContributingFactors.DateEnd,
		BehavioralRiskFactors.BehavioralFactor,
		BehavioralRiskFactors.Severity,
		BehavioralRiskFactorDateStart = BehavioralRiskFactors.DateStart,
		BehavioralRiskFactorDateEnd = BehavioralRiskFactors.DateEnd,
		LifeEvents.LifeEvent,
		LifeEventStartDate = LifeEvents.DateStart,
		LifeEventEndDate = LifeEVents.DateEnd,
		DIets.DietCatetory,
		Diets.FoodType,
		AvoidInDiet = Diets.AvoidedDietYN,
		VISPDATS.SPDAT_Type,
		VISPDATS.SPDAT_Date,
		VISPDATS.ServiceProvider,
		VISPDATS.PreScreenPeriod,
		--VISPDATS.QuestionE,
		--VISPDATS.DescriptionE,
		--VISPDATS.ScoreValue,
		VISPDATS.TotalScore


FROM 
	vw_ClientsServices
	INNER JOIN HIFIS_Clients ON vw_ClientsServices.ClientID = HIFIS_Clients.ClientID
	INNER JOIN HIFIS_People ON HIFIS_Clients.PersonID = HIFIS_People.PersonID
	INNER JOIN Consent ON vw_ClientsServices.ClientID = Consent.ClientID
	LEFT OUTER JOIN HIFIS_CountryTypes ON HIFIS_CLients.CountryOfBirth = HIFIS_CountryTypes.ID
	LEFT OUTER JOIN HIFIS_ProvinceTypes ON HIFIS_CLients.ProvinceOfBirth = HIFIS_ProvinceTypes.ID
	LEFT OUTER JOIN HIFIS_CityTypes ON HIFIS_Clients.CityOfBirth = HIFIS_CityTypes.CityTypeID
	LEFT OUTER JOIN HIFIS_HairColorTypes ON HIFIS_Clients.HairColorTypeID = HIFIS_HairColorTypes.ID
	LEFT OUTER JOIN HIFIS_EyeColorTypes ON HIFIS_CLients.EyeColorTypeID = HIFIS_EyeColorTypes.ID
	LEFT OUTER JOIN HIFIS_GeoRegionTypes ON HIFIS_People.GeoRegionTypeID = HIFIS_GeoRegionTypes.ID
	LEFT OUTER JOIN HIFIS_ClientEducationLevels ON vw_ClientsServices.ClientID = HIFIS_ClientEducationLevels.ClientID
	LEFT OUTER JOIN HIFIS_EducationLevelTypes ON HIFIS_ClientEducationLevels.EducationLevelTypeID = HIFIS_EducationLevelTypes.ID
	LEFT OUTER JOIN HousingPlacements ON vw_ClientsServices.ClientID = HousingPlacements.PrimaryClientID
	LEFT OUTER JOIN Families ON HIFIS_People.PersonID = Families.PersonID
	LEFT OUTER JOIN HealthIssues ON vw_ClientsServices.ClientID = HealthIssues.ClientID
	LEFT OUTER JOIN Incomes ON vw_ClientsServices.ClientID = Incomes.ClientID
	LEFT OUTER JOIN Expenses ON vw_ClientsServices.ClientID = Expenses.ClientID
	LEFT OUTER JOIN WatchConcerns ON vw_ClientsServices.ClientID = WatchConcerns.ClientID
	LEFT OUTER JOIN ContributingFactors ON vw_ClientsServices.ClientID = ContributingFactors.ClientID
	LEFT OUTER JOIN BehavioralRiskFactors ON vw_ClientsServices.ClientID = BehavioralRiskFactors.ClientID
	LEFT OUTER JOIN LifeEvents ON HIFIS_People.PersonID = LifeEvents.PersonID
	LEFT OUTER JOIN Medications ON vw_ClientsServices.ClientID = Medications.ClientID
	LEFT OUTER JOIN Diets ON vw_ClientsServices.ClientID = Diets.clientID
	LEFT OUTER JOIN VISPDATS ON vw_ClientsServices.ClientID = VISPDATS.ClientID
WHERE 
	vw_ClientsServices.OrganizationID IN (SELECT OrganizationID FROM HIFIS_Organizations WHERE ClusterID = 16)  -- no training clients

--AND ExpenseType LIKE '%Utili%'
--and Families.GroupID IS NOT NULL
--and Families.GroupID = 2964

--ORDER BY FamilyID,PeopleRelationshipTypeID --DateStart