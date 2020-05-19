-- Azure ML Client Export

USE HIFIS
GO

;WITH Consent AS(

SELECT * FROM
 (
SELECT 
			ClientID, 
			Con.ConsentID, 
			ConsentTypeID, 
			ORgID,
			ServiceProvider = orgs.Name,
			ConsentType = ConTypes.NameE, 
			ExpiryDate = min(ExpiryDate),
			Row# = ROW_NUMBER() OVER(PARTITION BY ClientID order by Case when ConsentTypeID != 2 then 1 else 2 end)
        FROM 
			HIFIS_Consent as Con
			INNER JOIN HIFIS_Consent_ServiceProviders as CSP ON Con.ConsentID = CSP.ConsentID
			INNER JOIN HIFIS_ConsentTypes as ConTypes ON Con.ConsentTypeID = ConTypes.ID
			INNER JOIN HIFIS_Organizations as Orgs ON CSP.OrgID = Orgs.OrganizationID
        WHERE 
			(GETDATE() BETWEEN Con.StartDate AND Con.ExpiryDate OR (GETDATE() >= Con.StartDate AND Con.ExpiryDate IS NULL))
        GROUP BY 
			ClientID,ConsentTypeID, Con.ConsentID,Orgid,NameE,ORgs.Name
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
		INNER JOIN HIFIS_Services b on b.ServiceID = a.ServiceID
		INNER JOIN HIFIS_Organizations c on c.OrganizationID = b.OrganizationID
	WHERE
	c.ClusterID <> 16
) t
WHERE Row = 1
AND MovedInDate IS NOT NULL
)
,
Families AS
(

	SELECT  
    pg.PeopleGroupID,
    pg.[GroupID], 
    pg.[DateEnd], 
    pg.[DateStart], 
    pg.[GroupRoleTypeID], 
    pg.[GroupHeadYN], 
    pg.[ServiceFee], 
    pg.[EmergencyContactYN], 
    pg.[PeopleRelationshipTypeID], 
    pg.[HifisRowVersion], 
    pg.[PersonID], 
    pg.[CreatedDate], 
    pg.[LastUpdatedDate], 
    pg.[LastUpdatedBy], 
    pg.[CreatedBy],
	prt.NameE as RelationshipType
--INTO 
--	#FAMILIES
FROM
	HIFIS_People_Groups pg
    INNER JOIN HIFIS_Groups g ON g.GroupID = pg.GroupID
	INNER JOIN HIFIS_PeopleRelationshipTypes prt ON prt.id = pg.PeopleRelationshipTypeID
	
WHERE 
( EXISTS 
		(	-- role type 11 = clients
			SELECT 
			1 AS [C1]
			FROM HIFIS_People_PeopleRoles ppr
			WHERE ([PersonID] = ppr.PersonID) --HIFIS_People_PeopleRoles.PersonID) 
			AND (11 = ppr.PeopleRoleTypeID))
			) 
	
	AND 
	(
		(pg.DateEnd IS NULL) 
		OR 
		(pg.DateEnd > GetDAte())
	) 
	AND 
	(
		(g.DateEnd IS NULL) 
		OR 
		(g.DateEnd > GetDate())
	) 
	AND pg.GroupRoleTypeID <> 9
	




),
Incomes
AS
(
		SELECT * FROM
		(
			SELECT
				--ROW_NUMBER() OVER (PARTITION BY HIFIS_ClientINComes.clientID,HIFIS_ClientIncomes.IncomeTypeID ORDER BY DateStart DESC) as RowNumber,
				ci.ClientID,
				IncomeTypeID,
				ClientIncomeID,
				it.NameE,
				MonthlyAmount,
				DateStart,
				DateEnd
				
			FROM 
				HIFIS_ClientIncomes ci
				INNER JOIN HIFIS_IncomeTypes it ON it.ID = ci.incomeTypeID
			WHERE 
				ci.createdDate IS NULL
			--AND
			--	ClientIncomeID NOT IN (Select ClientIncomeID FROM CTE)

		) t

),

Expenses AS
(
	SELECT 
	ce.ClientExpenseID,
	ce.ClientID,
	ce.ExpenseTypeID,
	ExpenseType = et.NameE,
	ce.DateStart,
	ce.DateEnd,
	ce.PayFrequencyTypeID,
	ExpenseFrequency = pft.NameE,
	ce.ExpenseAmount,
	ce.IsEssentialYN

	
	 FROM 
	HIFIS_ClientExpenses ce
	INNER JOIN HIFIS_ExpenseTypes et ON et.ID = ce.ExpenseTypeID 
	INNER JOIN HIFIS_PayFrequencyTypes pft ON pft.ID = ce.PayFrequencyTypeID

),

HealthIssues AS
(
SELECT 
hi.HealthIssueID,
hi.ClientID,
hi.HealthIssueTypeID,
HealthIssue = hit.NameE,
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
hm.MedicationName
 FROM
HIFIS_HealthIssues hi
INNER JOIN HIFIS_HealthIssueTypes hit ON hit.ID = hi.HealthIssueTypeID --HIFIS_HealthIssues.HealthIssueTypeID = HIFIS_HealthIssueTypes.ID
LEFT OUTER JOIN HIFIS_Medications hm ON hm.HealthIssueId = hi.HealthIssueID --HIFIS_HealthIssues.HealthIssueID = HIFIS_Medications.HealthIssueID

),

AssestsandLiabilities as
(
	select 
la.LiabilityOrAssetID,
la.ClientID,
Type='Asset',
la.DateStart,
la.DateEnd,
la.Description,
la.Amount,
ast.NameE

 FROM 
HIFIS_LiabilitiesOrAssests la
LEFT OUTER JOIN HIFIS_AssetTypes  ast ON ast.ID = la.AssetTypeID -- HIFIS_LiabilitiesOrAssests.AssetTypeID = HIFIS_AssetTypes.ID
WHERE AssetTypeID IS NOT NULL

UNION

select 
la.LiabilityOrAssetID,
la.ClientID,
Type='Liability',
la.DateStart,
la.DateEnd,
la.Description,
la.Amount,
lt.NameE
FROM 
HIFIS_LiabilitiesOrAssests la
LEFT OUTER JOIN HIFIS_LiabilityTypes lt ON lt.ID = la.AssetTypeID --HIFIS_LiabilitiesOrAssests.AssetTypeID = HIFIS_LiabilityTypes.ID
WHERE 
 LiabilityTypeID IS NOT NULL

),
WatchConcerns AS
(
	SELECT * FROM 
	(
	SELECT
		Row# = ROW_NUMBER() OVER (PARTITION BY cwc.ClientID ORDER BY DateStart DESC) ,
		cwc.ClientWatchConcernID,
		cwc.ClientID,
		cwc.WatchConcernTypeID,
		WatchConcern = wct.NameE,
		cwc.DateStart,
		cwc.DateEnd,
		cwc.Comments
	FROM 
		HIFIS_Client_WatchConcerns cwc
		INNER JOIN HIFIS_WatchConcernTypes wct ON  wct.ID = cwc.WatchConcernTypeID 
		--HIFIS_Client_WatchConcerns.WatchConcernTypeID = HIFIS_WatchConcernTypes.ID
	WHERE ID = 1000
) t
WHERE ROW#=1
	
),
ContributingFactors AS
(
	SELECT 
cf.ClientContributingFactorID,
cf.ClientID,
ContributingFactor = cft.NameE,
cf.DateStart,
cf.DateEnd

 FROM HIFIS_Client_ContributingFactor cf
INNER JOIN HIFIS_ContributingFactorTypes cft ON cft.ID = cf.ContributingTypeID
-- HIFIS_Client_ContributingFactor.ContributingTypeID = HIFIS_ContributingFactorTypes.ID
),
BehavioralRiskFactors AS
(

	SELECT 
	ClientBehaviouralFactorID,
	ClientID,
	BehavioralFactor = bft.NameE,
	Severity = pt.NameE,
	DateStart,
	DateEnd
	FROM HIFIS_Client_BehaviouralFactor cbf
	INNER JOIN HIFIS_BehaviouralFactorTypes bft ON bft.ID = cbf.BehavioralTypeID -- HIFIS_Client_BehaviouralFactor.BehavioralTypeID = HIFIS_BehaviouralFactorTypes.ID
	INNER JOIN HIFIS_ProbabilityTypes pt ON pt.ID = cbf.probabilityTypeID 
	-- HIFIS_Client_BehaviouralFactor.ProbabilityTypeID = HIFIS_ProbabilityTypes.ID


),
LifeEvents AS
(
	SELECT 
PeopleLifeEventID,
PersonID,
let.ID,
LifeEvent = let.NameE,
DateStart,
DateEnd
FROM
HIFIS_People_LifeEvents le
INNER JOIN HIFIS_LifeEventsTypes let ON let.ID = le.lifeEventTypeID
-- ON HIFIS_People_LifeEvents.LifeEventTypeID = HIFIS_LifeEventsTypes.ID
WHERE le.LifeEventTypeID <> 1005

)
,VISPDATS AS
	

(	
	SELECT  
			 cli.ClientID,
			 ppl.CurrentAge,
			 SPDAT_Type = sit.NameE,
			 PreScreenPeriod = psp.NameE,
			 AssessmentPeriod = apt.NameE,
			 
			 SPDAT_Date = si.StartDateTime,
			 si.LastUpdatedDate,
			 orgs.OrganizationID,
			 orgs.Name as ServiceProvider,
			 --Questions.AssessmentQuestionID,
			 --Questions.QuestionE,
			 --Questions.DescriptionE,
			 --QA.ScoreValue,
			 ss.TotalScore
			--RowNumber = ROW_NUMBER() OVER(PARTITION BY HIFIS_Clients.clientID ORDER BY HIFIS_Clients.ClientID,HIFIS_SPDAT_Intake.LastUpdatedDate DESC)
		FROM            
			HIFIS_SPDAT_Intake si
			INNER JOIN HIFIS_SPDAT_ScoringSummary ss on ss.intakeID = si.IntakeID
			-- ON HIFIS_SPDAT_Intake.IntakeID = HIFIS_SPDAT_ScoringSummary.IntakeID 
			INNER JOIN HIFIS_Services serv ON  serv.ServiceID = si.ServiceID
			--HIFIS_SPDAT_Intake.ServiceID = HIFIS_Services.ServiceID 
			INNER JOIN HIFIS_Client_Services cserv ON cserv.serviceID = serv.ServiceID 
			-- HIFIS_Services.ServiceID = HIFIS_Client_Services.ServiceID
			INNER JOIN HIFIS_Clients cli ON cli.ClientID = cserv.ClientID
			--HIFIS_Client_Services.ClientID = HIFIS_Clients.ClientID 
			INNER JOIN HIFIS_People ppl ON ppl.personID = cli.personID 
			--HIFIS_CLients.PersonID = HIFIS_People.PersonID 
			INNER JOIN HIFIS_SPDAT_IntakeTypes sit ON sit.ID = si.IntakeTYpe 
			--HIFIS_SPDAT_Intake.IntakeType = HIFIS_SPDAT_IntakeTypes.ID 
			INNER JOIN HIFIS_ORganizations orgs ON orgs.OrganizationID = serv.OrganizationID
			--HIFIS_Services.OrganizationID = HIFIS_Organizations.OrganizationID 
			LEFT OUTER JOIN	HIFIS_SPDAT_AssessmentPeriodTypes apt on apt.ID = si.AssessmentPeriodTypeID
			-- ON HIFIS_SPDAT_Intake.AssessmentPeriodTypeID = HIFIS_SPDAT_AssessmentPeriodTypes.ID 
			LEFT OUTER JOIN HIFIS_SPDAT_PreScreenPeriodTypes psp ON psp.ID = si.PreScreenPeriodTypeID
			-- HIFIS_SPDAT_Intake.PreScreenPeriodTypeID = HIFIS_SPDAT_PreScreenPeriodTypes.ID
			--INNER JOIN HIFIS_SPDAT_Intake_QuestionsAnswered QA ON HIFIS_SPDAT_Intake.IntakeID = QA.IntakeID
			--INNER JOIN HIFIS_SPDAT_AssessmentQuestions Questions ON QA.AssessmentQuestionID = Questions.AssessmentQuestionID
		WHERE 
			TOTALSCORE IS NOT NULL
		--	TotalScore > 8
		AND 
			cli.ClientStateTypeID = 1
		AND 
			--orgs.OrganizationID IN (SELECT OrganizationID FROM HIFIS_Organizations WHERE ClusterID = 16)
			orgs.ClusterID = 16
		AND
			sit.NameE LIKE '%VI%'



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
cd.ClientDietID,
cd.ClientID,
DietCatetory = dct.NameE,
FoodType = dfi.NameE,
cd.AvoidedDietYN

 FROM
HIFIS_ClientDiets cd
INNER JOIN HIFIS_DietCategoryTypes dct ON dct.ID = cd.DietCategoryTypeID
-- HIFIS_ClientDiets.DietCategoryTypeID = HIFIS_DietCategoryTypes.ID
INNER JOIN HIFIS_DietFoodItemTypes dfi ON dfi.ID = cd.DietFoodItemTypeID
-- HIFIS_ClientDiets.DietFoodItemTypeID = HIFIS_DietFoodItemTypes.ID

)







SELECT 

	Serv.ServiceID,
	cli.ClientID,
	Consent.ConsentType,
	FamilyID = Families.GroupID,
	--Families.PeopleRelationshipTypeID,
	Families.RelationshipType,
	ppl.DOB,
	ppl.CurrentAge,
	--CurrentAge = DATEDIFF(Day,HIFIS_People.DOB,GetDate()) / 365.2422,
	Gender = gt.NameE,--  cbas.Gender_En, --Gender_En,
	AboriginalIndicator = ait.NameE, --serv.AboriginalIndicator_En,
	Citizenship = ctz.NameE, --serv.Citizenship_En,
	VeteranStatus = vst.NameE, --serv.VeteranStatus_En,
	CountryOfBirth = ctry.NameE,
	ProvinceOfBirth = COALESCE(Prov.NameE,ProvinceFreeText),
	CityOfBirth = COALESCE(city.NameE,CityFreeText),
	GeoRegion = geo.NameE,
	HairColour = hct.NameE,
	EyeColour = eye.NameE,
	ClientHeightCM = ClientHeight,
	ClientWeightKG = ClientWeight,
	ServiceType = ServTypes.NameE, -- serv.ServiceType_EN
	orgs.OrganizationID,
	Orgs.Name as OrganizationName,
	ServiceStartDate = serv.DateStart,
	ServiceEndDate = serv.DateEnd,
	ReasonForService = rfs.NameE, --serv.ReasonForService_En,
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
	EducationLevel = edu.NameE,
	EducationStartDate = cel.DateStart,
	EducationEndDate = cel.DateEnd,
	HealthIssues.HealthIssue,
	HealthIssues.DiagnosedYN,
	HealthIssues.SelfReportedYN,
	HealthIssues.SuspectedYN,
	HealthIssueStart = HealthIssues.DateFrom,
	HealthIssueEnd = HealthIssues.DateTo,
	HealthIssueMedicationName = HealthIssues.MedicationName,
	OtherMedications = Medications.MedicationName,
	WatchConcerns.WatchConcern,
	WatchConcerns.DateStart as WatchConcernDateStart,
	WatchConcerns.DateEnd as WatchConcernDateEnd,
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
	VISPDATS.TotalScore
FROM 
	HIFIS_Services serv
	--vw_ClientsServices Serv
	INNER JOIN  HIFIS_Client_Services cserv ON cserv.ServiceID = serv.ServiceID
	INNER JOIN HIFIS_Clients cli ON cli.ClientID = cserv.ClientID
	INNER JOIN HIFIS_People ppl ON ppl.PersonID = cli.PersonID
	INNER JOIN HIFIS_Organizations orgs ON orgs.OrganizationID = serv.OrganizationID
	--INNER JOIN vw_ClientBasics cbas on cbas.ClientID = cli.clientID
	INNER JOIN HIFIS_ServiceTypes servTypes ON ServTypes.ID = serv.ServiceTypeID
	LEFT OUTER JOIN HIFIS_ReasonForServiceTypes rfs on rfs.ID = serv.ReasonForServiceTYpeID
	INNER JOIN HIFIS_AboriginalIndicatorTypes ait ON  ait.ID = cli.AboriginalIndicatorID
	INNER JOIN HIFIS_CitizenshipTypes ctz ON ctz.ID = cli.CitizenshipTypeID
	INNER JOIN HIFIS_VeteranStatesTypes vst ON vst.ID = cli.VeteranStateID
	INNER JOIN HIFIS_GenderTypes gt ON gt.ID = ppl.GenderTypeID
	INNER JOIN Consent ON cli.ClientID = Consent.ClientID
	LEFT OUTER JOIN HIFIS_CountryTypes ctry ON ctry.ID = cli.CountryOfBirth --HIFIS_CLients.CountryOfBirth = HIFIS_CountryTypes.ID
	LEFT OUTER JOIN HIFIS_ProvinceTypes prov ON prov.id = cli.ProvinceOfBirth --HIFIS_CLients.ProvinceOfBirth = HIFIS_ProvinceTypes.ID
	LEFT OUTER JOIN HIFIS_CityTypes city ON city.CityTypeID = cli.CityOfBirth --HIFIS_Clients.CityOfBirth = HIFIS_CityTypes.CityTypeID
	LEFT OUTER JOIN HIFIS_HairColorTypes hct ON hct.ID = cli.HairColorTypeID --HIFIS_Clients.HairColorTypeID = HIFIS_HairColorTypes.ID
	LEFT OUTER JOIN HIFIS_EyeColorTypes eye ON eye.ID = cli.EyeColorTypeID -- HIFIS_CLients.EyeColorTypeID = HIFIS_EyeColorTypes.ID
	LEFT OUTER JOIN HIFIS_GeoRegionTypes geo ON geo.ID = ppl.GeoRegionTypeID -- HIFIS_People.GeoRegionTypeID = HIFIS_GeoRegionTypes.ID
	LEFT OUTER JOIN HIFIS_ClientEducationLevels cel ON cel.clientID = cli.ClientID --  vw_ClientsServices.ClientID = HIFIS_ClientEducationLevels.ClientID
	LEFT OUTER JOIN HIFIS_EducationLevelTypes edu ON edu.ID = cel.EducationLevelTypeID -- HIFIS_ClientEducationLevels.EducationLevelTypeID = HIFIS_EducationLevelTypes.ID
	LEFT OUTER JOIN HousingPlacements ON HousingPlacements.PrimaryClientID = cli.ClientID -- = HousingPlacements.PrimaryClientID
	LEFT OUTER JOIN Families ON Families.PersonID = ppl.personID
	LEFT OUTER JOIN HealthIssues ON HealthIssues.ClientID = cli.ClientID 
	LEFT OUTER JOIN Incomes ON Incomes.ClientID = cli.ClientID
	LEFT OUTER JOIN Expenses ON Expenses.ClientID = cli.ClientID
	LEFT OUTER JOIN WatchConcerns ON WatchConcerns.ClientID = cli.ClientID
	LEFT OUTER JOIN ContributingFactors ON ContributingFactors.ClientID = cli.ClientID
	LEFT OUTER JOIN BehavioralRiskFactors ON BehavioralRiskFactors.ClientID = cli.ClientID
	LEFT OUTER JOIN LifeEvents ON LifeEvents.PersonID = ppl.PersonID
	LEFT OUTER JOIN Medications ON Medications.ClientID = cli.ClientID
	LEFT OUTER JOIN Diets ON Diets.clientID = cli.ClientID
	LEFT OUTER JOIN VISPDATS ON VISPDATS.ClientID = cli.ClientID
WHERE 
	orgs.ClusterID = 16 -- HIFIS Shared Cluster
	--serv.OrganizationID  IN (SELECT OrganizationID FROM HIFIS_Organizations WHERE ClusterID = 16)
	

ORDER BY ClientID 

