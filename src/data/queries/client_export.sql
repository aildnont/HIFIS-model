;WITH Consent AS
(
SELECT 
	* 
FROM
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
			AND Orgs.ClusterID = 16
        WHERE 
			(GETDATE() BETWEEN Con.StartDate AND Con.ExpiryDate OR (GETDATE() >= Con.StartDate AND Con.ExpiryDate IS NULL))
        GROUP BY 
			ClientID,ConsentTypeID, Con.ConsentID,Orgid,NameE,ORgs.Name
		) as abc
    WHERE 
		Row# = 1
),
HousingPlacements AS
(
--- HOUSE PLACEMENTS ------------------------------------------------
SELECT 
	HousePlacementID,
	PrimaryClientID,
	MovedInDate,
	PlacementEndedYN,
	FollowUpCompleteYN,
	DateEnd
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
		PlacementEndedYN,
		FollowUpCompleteYN,
		b.DateEnd
	FROM
		HIFIS_HousePlacements a
		INNER JOIN HIFIS_Services b on b.ServiceID = a.ServiceID
		INNER JOIN HIFIS_Organizations c on c.OrganizationID = b.OrganizationID
	WHERE
	c.ClusterID = 16
	AND
	a.PlacementEndedYN = 'N'
	AND
	MovedInDate IS NOT NULL
	
		
) t
WHERE Row = 1
),
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
ClientBasics AS
(
--- Client Basics
----------------------------------------------------

SELECT
	vw_ClientBasics.ClientID,
	HIFIS_Clients.PersonID,
	Consent.ConsentType,
	Families.GroupID as FamilyID,
	Families.RelationshipType,
	HIFIS_People.DOB,
	vw_ClientBasics.CurrentAge,
	Gender_En as Gender,
	AboriginalIndicator_En as AboriginalIndicator,
	Citizenship_En as Citizenship,
	VeteranStatus_En as VeteranStatus,
	HIFIS_CountryTypes.NameE as CountryOfBirth,
	HIFIS_ProvinceTypes.NameE as ProvinceOfBirth,
	HIFIS_CityTypes.NameE as CityOfBirth,
	HIFIS_EyeColorTypes.NameE as EyeColour,
	HIFIS_HairColorTypes.NameE as HairColour,
	HIFIS_Clients.ClientHeight as ClientHeightCM,
	HIFIS_Clients.ClientWeight as ClientWeightKG,
	CASE WHEN HousingPlacements.MovedInDate IS NOT NULL THEN 'Y' ELSE 'N' END as InHousing,
	HousingPlacements.MovedInDate
FROM 
	vw_ClientBasics
	INNER JOIN Consent ON vw_ClientBasics.ClientID = Consent.ClientID
	INNER JOIN HIFIS_Clients ON vw_ClientBasics.ClientID = HIFIS_Clients.ClientID
	INNER JOIN HIFIS_People ON HIFIS_People.PersonID = HIFIS_Clients.PersonID
	LEFT OUTER JOIN HIFIS_CountryTypes ON HIFIS_CLients.CountryOfBirth = HIFIS_CountryTypes.ID
	LEFT OUTER JOIN HIFIS_ProvinceTypes ON HIFIS_Clients.ProvinceOfBirth = HIFIS_ProvinceTypes.ID
	LEFT OUTER JOIN HIFIS_CityTypes ON HIFIS_Clients.CityOfBirth = HIFIS_CityTypes.CityTypeID
	LEFT OUTER JOIN HIFIS_EyeColorTypes ON HIFIS_Clients.EyeColorTypeID = HIFIS_EyeColorTypes.ID
	LEFT OUTER JOIN HIFIS_HairColorTypes ON HIFIS_Clients.HairColorTypeID = HIFIS_HairColorTypes.ID
	LEFT OUTER JOIN HousingPlacements ON vw_ClientBasics.ClientID = HousingPlacements.PrimaryClientID
	LEFT OUTER JOIN Families ON HIFIS_Clients.PersonID = Families.PersonID

),
Services AS
(

SELECT
	clientID,
	ServiceID,
	ServiceType_En as ServiceType,
	DateStart,
	DateEnd,
	ReasonForService_En as ReasonForService,
	OrganizationName
FROM
	vw_ClientsServices
WHERE
	EXISTS (SELECT 1 FROM ClientBasics WHERE vw_ClientsServices.ClientID = ClientBasics.ClientID)
),
Incomes AS
(


--- incomes ---------------------------------------------------------------------------------------------------------------

SELECT 
	* 

FROM
		(
			SELECT
				ci.ClientID,
				IncomeTypeID,
				ClientIncomeID,
				it.NameE,
				MonthlyAmount,
				DateStart,
				DateEnd,
				ci.OwnerOrgID
				
			FROM 
				HIFIS_ClientIncomes ci
				INNER JOIN HIFIS_IncomeTypes it ON it.ID = ci.incomeTypeID
			WHERE 
				ci.createdDate IS NULL
			
		) t

),
Expenses AS
(

--- EXPENSES ---------------------------------------------------------------

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

--- HEALTH ISSUES --------------------------------------------------------------------------------------------

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
Medications AS
(
-- MEDICATIONS -------------------------------------------------------------------------------------------------------------------------

SELECT 
	medicationID,
	ClientID,
	MedicationName,
	Dosage,
	DateStart,
	DateEnd
FROM
	HIFIS_Medications 
WHERE 
	HealthIssueID IS NULL
),


--- BY NAME LIST WATCH CONCERNS (ID = 10000) ---------------------------------------------------------------------------------

/*  NO LONGER USED BY HP BUT KEEPING IN CASE
SELECT *
INTO
	##WatchConcerns
FROM 
	(
	SELECT
		Row## = ROW_NUMBER() OVER (PARTITION BY cwc.ClientID ORDER BY DateStart DESC) ,
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

*/ 
--CONTRIBUTING FACTORS ------------------------------------------------------------------------------
ContributingFactors AS
(
SELECT 
	cf.ClientContributingFactorID,
	cf.ClientID,
	ContributingFactor = cft.NameE,
	cf.DateStart,
	cf.DateEnd
FROM 
	HIFIS_Client_ContributingFactor cf
	INNER JOIN HIFIS_ContributingFactorTypes cft ON cft.ID = cf.ContributingTypeID

),

BehavioralRiskFactors as
(
--- BEHAVIORAL RISK FACTORS -----------------------------------------------------------------------

SELECT 
	ClientBehaviouralFactorID,
	ClientID,
	BehavioralFactor = bft.NameE,
	Severity = pt.NameE,
	DateStart,
	DateEnd

FROM 
	HIFIS_Client_BehaviouralFactor cbf
	INNER JOIN HIFIS_BehaviouralFactorTypes bft ON bft.ID = cbf.BehavioralTypeID
	INNER JOIN HIFIS_ProbabilityTypes pt ON pt.ID = cbf.probabilityTypeID 

),
LifeEvents AS
(

--- LIFE EVENTS -----------------------------------------------------------------------------------------------

SELECT 
	PeopleLifeEventID,
	le.PersonID,
	Clients.ClientID,
	let.ID,
	LifeEvent = let.NameE,
	DateStart,
	DateEnd

FROM
	HIFIS_People_LifeEvents le
	INNER JOIN HIFIS_LifeEventsTypes let ON let.ID = le.lifeEventTypeID
	INNER JOIN HIFIS_Clients clients ON le.PersonID = Clients.PersonID
WHERE 
	le.LifeEventTypeID <> 1005

),
Diets AS
(

-- DIETS -----------------------------------------------------------------------------------------------------

SELECT 
	cd.ClientDietID,
	cd.ClientID,
	DietCatetory = dct.NameE,
	FoodType = dfi.NameE,
	cd.AvoidedDietYN,
	cd.CreatedDate
FROM
	HIFIS_ClientDiets cd
	INNER JOIN HIFIS_DietCategoryTypes dct ON dct.ID = cd.DietCategoryTypeID
	INNER JOIN HIFIS_DietFoodItemTypes dfi ON dfi.ID = cd.DietFoodItemTypeID
),
VISPDATS AS
(

--  VI-SPDATS -------------------------------------------------------------------------------------------------

SELECT  
	cli.ClientID,
	si.IntakeID,
	ppl.CurrentAge,
	SPDAT_Type = sit.NameE,
	PreScreenPeriod = psp.NameE,
	AssessmentPeriod = apt.NameE,
	SPDAT_Date = si.StartDateTime,
	si.LastUpdatedDate,
	orgs.OrganizationID,
	orgs.Name as ServiceProvider,
	ss.TotalScore
FROM            
	HIFIS_SPDAT_Intake si
	INNER JOIN HIFIS_SPDAT_ScoringSummary ss on ss.intakeID = si.IntakeID
	INNER JOIN HIFIS_Services serv ON  serv.ServiceID = si.ServiceID
	INNER JOIN HIFIS_Client_Services cserv ON cserv.serviceID = serv.ServiceID 
	INNER JOIN HIFIS_Clients cli ON cli.ClientID = cserv.ClientID
	INNER JOIN HIFIS_People ppl ON ppl.personID = cli.personID 
	INNER JOIN HIFIS_SPDAT_IntakeTypes sit ON sit.ID = si.IntakeTYpe 
	INNER JOIN HIFIS_ORganizations orgs ON orgs.OrganizationID = serv.OrganizationID
	LEFT OUTER JOIN	HIFIS_SPDAT_AssessmentPeriodTypes apt on apt.ID = si.AssessmentPeriodTypeID
	LEFT OUTER JOIN HIFIS_SPDAT_PreScreenPeriodTypes psp ON psp.ID = si.PreScreenPeriodTypeID
WHERE 
	TOTALSCORE IS NOT NULL
AND 
	cli.ClientStateTypeID = 1
AND 
	orgs.ClusterID = 16
AND
	sit.NameE LIKE '%VI%'

),
-- EDUCATION LEVELS ------------------------------------------------------------------------------------------------------
EducationLevels AS
(

SELECT 
	A.ClientID,
	B.NameE as EducationLevel,
	A.DateStart,
	A.DateEnd
FROM
	HIFIS_ClientEducationLevels A
	INNER JOIN HIFIS_EducationLevelTypes B  ON A.EducationLevelTypeID = B.ID
),

-- SERVICE RESTRICTIONS  ----------------------------------------------------------------------------------------------

ClientBarredPeriods AS
(
	SELECT
		A.ClientBarredPeriodID,
		A.ClientID,
		A.DateStart,
		A.DateEnd,
		B.NameE as Reason,
		Orgs.name as OrganizationName

	FROM 
		HIFIS_Client_Barred_Periods A
		INNER JOIN HIFIS_ReasonBarredTypes B ON A.ReasonBarredTypeID = B.ID
		INNER JOIN HIFIS_Organizations_Client_Barred_Periods C  ON A.ClientBarredPeriodID = C.ClientBarredPeriodID
		INNER JOIN HIFIS_Organizations Orgs ON C.OrganizationID = orgs.OrganizationID
	WHERE 
		CAST(A.DateStart as DAte) >= '2017-07-01'  -- there are historical restrictions that have been imported into HIFIS.  
													--only use data created in HIFIS applicaiton. HIFIS use started in July 2017
	 AND EXISTS (
		SELECT 1 FROM Consent WHERE A.ClientID = Consent.ClientID

		)
)
,
BarredModules AS
(

	SELECT
		A.ClientBarredPeriodID,
		C.NameE as Module
	FROM
		HIFIS_Client_Barred_Periods A
		INNER JOIN HIFIS_Barred_Modules B ON A.ClientBarredPeriodID = B.ClientBarredPeriodID
		INNER JOIN HIFIS_ServiceRestrictionsModuleTypes C ON B.ServiceRestrictionModuleTypeID = C.ID
),
ServiceRestrictions AS
(

	SELECT 
	ClientBarredPeriodID,
	ClientID,
	DateStart,
	DateEnd,
	Reason,
	OrganizationName,
	STUFF(( SELECT '|' + Module
		FROM BarredModules t2
		WHERE t2.clientBarredPeriodID = t1.ClientBarredPeriodID
		FOR XML PATH('')),1,1,'')  as Modules
FROM 
	ClientBarredPeriods t1
)




-- MAIN  QUERY --------------------------------------------------------------------------------------------------


SELECT
	--COUNT(DISTINCT ClientID) as TotalClients
	*
FROM
(
SELECT
	ClientBasics.*,
	Services.ServiceID,
	Services.ServiceType,
	Services.DateStart,
	Services.DateEnd,
	Services.ReasonForService,
	Services.OrganizationName,
	NULL as IncomeType,
	NULL as MonthlyAmount,
	NULL AS ClientExpenseID,
	NULL AS ExpenseType,
	NULL AS ExpenseAmount,
	NULL AS ExpenseFrequency,
	NULL AS IsEssentialYN,
	NULL as EducationLevel,
	NULL as EducationStartDate,
	NULL as EducationEndDate,
	NULL as HealthIssueID,
	NULL AS HealthIssue,
	NULl as DiagnosedYN,
	NULL as SelfReportedYN,
	NULL as SuspectedYN,
	NULL as HealthIssuesMedicationName,
	NULL as OtherMedications,
	--NULL AS ClientWatchConcernID,
	--NULL AS WatchConcern,
	NULL AS clientContributingFactorID,
	NULL AS ContributingFactor,
	NULL AS ClientBehaviouralFactorID,
	NULL AS BehavioralFactor,
	NULL AS Severity,
	NULL AS PeopleLifeEventID,
	NULL as LifeEvent,
	NULL AS ClientDietID,
	NULL AS DietCatetory,
	NULL AS FoodType,
	NULL AS AvoidedDietYN,
	NULL as IntakeID,
	NULL as PreScreenPeriod,
	NULL as TotalScore,
	NULL AS ClientBarredPeriodID,
	NULL as Reason,
	NULL as Modules,
	NULL as ServiceRestricitonOrganizationName
FROM
	ClientBasics
	INNER JOIN Services ON ClientBasics.ClientID = Services.ClientID

	
	
UNION ALL

SELECT
	ClientBasics.*,
	NULL as ServiceID,
	'Income' as ServiceType,
	incomes.DateStart,
	incomes.DateEnd,
	incomes.NameE,
	NULL as OranizationName,
	incomes.NameE as IncomeType,
	incomes.MonthlyAmount,
	NULL AS ClientExpenseID,
	NULL AS ExpenseType,
	NULL AS ExpenseAmount,
	NULL AS ExpenseFrequency,
	NULL AS IsEssentialYN,
	NULL as EducationLevel,
	NULL as EducationStartDate,
	NULL as EducationEndDate,
	NULL as HealthIssueID,
	NULL AS HealthIssue,
	NULl as DiagnosedYN,
	NULL as SelfReportedYN,
	NULL as SuspectedYN,
	NULL as HealthIssuesMedicationName,
	NULL as OtherMedications,
	--NULL AS ClientWatchConcernID,
	--NULL AS WatchConcern,
	NULL AS clientContributingFactorID,
	NULL AS ContributingFactor,
	NULL AS ClientBehaviouralFactorID,
	NULL AS BehavioralFactor,
	NULL AS Severity,
	NULL AS PeopleLifeEventID,
	NULL as LifeEvent,
	NULL AS ClientDietID,
	NULL AS DietCatetory,
	NULL AS FoodType,
	NULL AS AvoidedDietYN,
	NULL as IntakeID,
	NULL as PreScreenPeriod,
	NULL as TotalScore,
	NULL AS ClientBarredPeriodID,
	NULL as Reason,
	NULL as Modules,
	NULL as ServiceRestricitonOrganizationName

FROM
	ClientBasics
	INNER JOIN incomes ON ClientBasics.ClientID = incomes.ClientID
	

UNION ALL

SELECT
	ClientBasics.*,
	NULL as ServiceID,
	'Expenses' as ServiceType,
	Expenses.DateStart,
	Expenses.DateEnd,
	NULL AS ReasonForService,
	NULL AS OrganizationName,
	NULL AS IncomeType,
	NULL as MonthlyAmount,
	Expenses.ClientExpenseID,
	Expenses.ExpenseType,
	Expenses.ExpenseAmount,
	Expenses.ExpenseFrequency,
	Expenses.IsEssentialYN,
	NULL as EducationLevel,
	NULL as EducationStartDate,
	NULL as EducationEndDate,
	NULL as HealthIssueID,
	NULL AS HealthIssue,
	NULl as DiagnosedYN,
	NULL as SelfReportedYN,
	NULL as SuspectedYN,
	NULL as HealthIssuesMedicationName,
	NULL as OtherMedications,
	--NULL AS ClientWatchConcernID,
	--NULL AS WatchConcern,
	NULL AS clientContributingFactorID,
	NULL AS ContributingFactor,
	NULL AS ClientBehaviouralFactorID,
	NULL AS BehavioralFactor,
	NULL AS Severity,
	NULL AS PeopleLifeEventID,
	NULL as LifeEvent,
	NULL AS ClientDietID,
	NULL AS DietCatetory,
	NULL AS FoodType,
	NULL AS AvoidedDietYN,
	NULL as IntakeID,
	NULL as PreScreenPeriod,
	NULL as TotalScore,
	NULL AS ClientBarredPeriodID,
	NULL as Reason,
	NULL as Modules,
	NULL as ServiceRestricitonOrganizationName
FROM
	ClientBasics
	INNER JOIN Expenses ON ClientBasics.ClientID = Expenses.ClientID




UNION ALL

SELECT
	ClientBasics.*,
	NULL as ServiceID,
	'Education' as ServiceType,
	EducationLevels.DateStart,
	EducationLevels.DateEnd,
	NULL AS ReasonForService,
	NULL AS OrganizationName,
	NULL AS IncomeType,
	NULL as MonthlyAmount,
	NULL AS ClientExpenseID,
	NULL AS ExpenseType,
	NULL AS ExpenseAmount,
	NULL AS ExpenseFrequency,
	NULL IsEssentialYN,
	EducationLevels.EducationLevel,
	NULL as EducationStartDate,
	NULL as EducationEndDate,
	NULL as HealthIssueID,
	NULL AS HealthIssue,
	NULl as DiagnosedYN,
	NULL as SelfReportedYN,
	NULL as SuspectedYN,
	NULL as HealthIssuesMedicationName,
	NULL as OtherMedications,
	--NULL AS ClientWatchConcernID,
	--NULL AS WatchConcern,
	NULL AS clientContributingFactorID,
	NULL AS ContributingFactor,
	NULL AS ClientBehaviouralFactorID,
	NULL AS BehavioralFactor,
	NULL AS Severity,
	NULL AS PeopleLifeEventID,
	NULL as LifeEvent,
	NULL AS ClientDietID,
	NULL AS DietCatetory,
	NULL AS FoodType,
	NULL AS AvoidedDietYN,
	NULL as IntakeID,
	NULL as PreScreenPeriod,
	NULL as TotalScore,
	NULL AS ClientBarredPeriodID,
	NULL as Reason,
	NULL as Modules,
	NULL as ServiceRestricitonOrganizationName
FROM
	ClientBasics
	INNER JOIN EducationLevels ON ClientBasics.ClientID = EducationLevels.ClientID

UNION ALL

SELECT
	ClientBasics.*,
	NULL as ServiceID,
	'HealthIssues' as ServiceType,
	HealthIssues.DateFrom as DateStart,
	HealthIssues.DateTo as DateEnd,
	NULL AS ReasonForService,
	NULL AS OrganizationName,
	NULL AS IncomeType,
	NULL as MonthlyAmount,
	NULL as ClientExpenseID,
	NULL AS ExpenseType,
	NULL AS ExpenseAmount,
	NULL AS ExpenseFrequency,
	NULL as IsEssentialYN,
	NULL as EducationLevel,
	NULL as EducationStartDate,
	NULL as EducationEndDate,
	HealthIssues.HealthIssueID,
	HealthIssues.HealthIssue,
	HealthIssues.DiagnosedYN,
	HealthIssues.SelfReportedYN,
	HealthIssues.SuspectedYN,
	HealthIssues.MedicationName as HealthIssuesMedicationName,
	NULL as OtherMedications,
	--NULL AS ClientWatchConcernID,
	--NULL AS WatchConcern,
	NULL AS clientContributingFactorID,
	NULL AS ContributingFactor,
	NULL AS ClientBehaviouralFactorID,
	NULL AS BehavioralFactor,
	NULL AS Severity,
	NULL AS PeopleLifeEventID,
	NULL as LifeEvent,
	NULL AS ClientDietID,
	NULL AS DietCatetory,
	NULL AS FoodType,
	NULL AS AvoidedDietYN,
	NULL as IntakeID,
	NULL as PreScreenPeriod,
	NULL as TotalScore,
	NULL AS ClientBarredPeriodID,
	NULL as Reason,
	NULL as Modules,
	NULL as ServiceRestricitonOrganizationName

FROM
	ClientBasics
	INNER JOIN HealthIssues ON ClientBasics.ClientID = HealthIssues.ClientID

UNION ALL

SELECT
	ClientBasics.*,
	NULL as ServiceID,
	'Medications' as ServiceType,
	Medications.DateStart,
	Medications.DateEnd,
	NULL AS ReasonForService,
	NULL AS OrganizationName,
	NULL AS IncomeType,
	NULL as MonthlyAmount,
	NULL as ClientExpenseID,
	NULL AS ExpenseType,
	NULL AS ExpenseAmount,
	NULL AS ExpenseFrequency,
	NULL as IsEssentialYN,
	NULL as EducationLevel,
	NULL as EducationStartDate,
	NULL as EducationEndDate,
	NULL as HealthIssueID,
	NULL as HealthIssue,
	NULL as DiagnosedYN,
	NULL as SelfReportedYN,
	NULL as SuspectedYN,
	NULL as HealthIssuesMedicationName,
	Medications.MedicationName as OtherMedications,
	--NULL AS ClientWatchConcernID,
	--NULL AS WatchConcern,
	NULL AS clientContributingFactorID,
	NULL AS ContributingFactor,
	NULL AS ClientBehaviouralFactorID,
	NULL AS BehavioralFactor,
	NULL AS Severity,
	NULL AS PeopleLifeEventID,
	NULL as LifeEvent,
	NULL AS ClientDietID,
	NULL AS DietCatetory,
	NULL AS FoodType,
	NULL AS AvoidedDietYN,
	NULL as IntakeID,
	NULL as PreScreenPeriod,
	NULL as TotalScore,
	NULL AS ClientBarredPeriodID,
	NULL as Reason,
	NULL as Modules,
	NULL as ServiceRestricitonOrganizationName

FROM
	ClientBasics
	INNER JOIN Medications ON ClientBasics.ClientID = Medications.ClientID



/*
UNION ALL
-- Watch concerns are commented out as they are no longer used by Homeless Prevention.
-- keeping code in case this changes again
SELECT
	ClientBasics.*,
	'Watch Concern' as ServiceType,
	WatchConcerns.DateStart,
	WatchConcerns.DateEnd,
	NULL AS ReasonForService,
	NULL AS OrganizationName,
	NULL AS IncomeType,
	NULL as MonthlyAmount,
	NULL as ClientExpenseID,
	NULL AS ExpenseType,
	NULL AS ExpenseAmount,
	NULL AS ExpenseFrequency,
	NULL as IsEssentialYN,
	NULL as EducationLevel,
	NULL as EducationStartDate,
	NULL as EducationEndDate,
	NULL as HealthIssueID,
	NULL as HealthIssue,
	NULL as DiagnosedYN,
	NULL as SelfReportedYN,
	NULL as SuspectedYN,
	NULL as HealthIssuesMedicationName,
	NULL as OtherMedications,
	WatchConcerns.ClientWatchConcernID,
	WatchConcerns.WatchConcern,
	NULL AS clientContributingFactorID,
	NULL AS ContributingFactor,
	NULL AS ClientBehaviouralFactorID,
	NULL AS BehavioralFactor,
	NULL AS Severity,
	NULL AS PeopleLifeEventID,
	NULL as LifeEvent,
	NULL AS ClientDietID,
	NULL AS DietCatetory,
	NULL AS FoodType,
	NULL AS AvoidedDietYN,
	NULL as IntakeID,
	NULL as PreScreenPeriod,
	NULL as TotalScore
FROM
	ClientBasics
	INNER JOIN WatchConcerns ON ClientBasics.ClientID = WatchConcerns.ClientID

*/ 


UNION ALL

SELECT
	ClientBasics.*,
	NULL as ServiceID,
	'Contributing Factor' as ServiceType,
	ContributingFactors.DateStart,
	ContributingFactors.DateEnd,
	NULL AS ReasonForService,
	NULL AS OrganizationName,
	NULL AS IncomeType,
	NULL as MonthlyAmount,
	NULL as ClientExpenseID,
	NULL AS ExpenseType,
	NULL AS ExpenseAmount,
	NULL AS ExpenseFrequency,
	NULL as IsEssentialYN,
	NULL as EducationLevel,
	NULL as EducationStartDate,
	NULL as EducationEndDate,
	NULL as HealthIssueID,
	NULL as HealthIssue,
	NULL as DiagnosedYN,
	NULL as SelfReportedYN,
	NULL as SuspectedYN,
	NULL as HealthIssuesMedicationName,
	NULL as OtherMedications,
	--NULL as ClientWatchConcernID,
	--NULL as WatchConcern,
	ContributingFactors.ClientContributingFactorID,
	ContributingFactors.ContributingFactor,
	NULL AS ClientBehaviouralFactorID,
	NULL AS BehavioralFactor,
	NULL AS Severity,
	NULL AS PeopleLifeEventID,
	NULL as LifeEvent,
	NULL AS ClientDietID,
	NULL AS DietCatetory,
	NULL AS FoodType,
	NULL AS AvoidedDietYN,
	NULL as IntakeID,
	NULL as PreScreenPeriod,
	NULL as TotalScore,
	NULL AS ClientBarredPeriodID,
	NULL as Reason,
	NULL as Modules,
	NULL as ServiceRestricitonOrganizationName
FROM
	ClientBasics
	INNER JOIN ContributingFactors ON ClientBasics.ClientID = ContributingFactors.ClientID




UNION ALL

SELECT
	ClientBasics.*,
	NULL as ServiceID,
	'Behavioral Risk Factor' as ServiceType,
	BehavioralRiskFactors.DateStart,
	BehavioralRiskFactors.DateEnd,
	NULL AS ReasonForService,
	NULL AS OrganizationName,
	NULL AS IncomeType,
	NULL as MonthlyAmount,
	NULL as ClientExpenseID,
	NULL AS ExpenseType,
	NULL AS ExpenseAmount,
	NULL AS ExpenseFrequency,
	NULL as IsEssentialYN,
	NULL as EducationLevel,
	NULL as EducationStartDate,
	NULL as EducationEndDate,
	NULL as HealthIssueID,
	NULL as HealthIssue,
	NULL as DiagnosedYN,
	NULL as SelfReportedYN,
	NULL as SuspectedYN,
	NULL as HealthIssuesMedicationName,
	NULL as OtherMedications,
	--NULL as ClientWatchConcernID,
	--NULL as WatchConcern,
	NULL as ClientContributingFactorID,
	NULL AS ContributingFactor,
	BehavioralRiskFactors.ClientBehaviouralFactorID,
	BehavioralRiskFactors.BehavioralFactor,
	BehavioralRiskFactors.Severity,
	NULL AS PeopleLifeEventID,
	NULL as LifeEvent,
	NULL AS ClientDietID,
	NULL AS DietCatetory,
	NULL AS FoodType,
	NULL AS AvoidedDietYN,
	NULL as IntakeID,
	NULL as PreScreenPeriod,
	NULL as TotalScore,
	NULL AS ClientBarredPeriodID,
	NULL as Reason,
	NULL as Modules,
	NULL as ServiceRestricitonOrganizationName
FROM
	ClientBasics
	INNER JOIN BehavioralRiskFactors ON ClientBasics.ClientID = BehavioralRiskFactors.ClientID


UNION ALL

SELECT
	ClientBasics.*,
	NULL as ServiceID,
	'Life Events' as ServiceType,
	lifeEvents.DateStart,
	lifeEvents.DateEnd,
	NULL AS ReasonForService,
	NULL AS OrganizationName,
	NULL AS IncomeType,
	NULL as MonthlyAmount,
	NULL as ClientExpenseID,
	NULL AS ExpenseType,
	NULL AS ExpenseAmount,
	NULL AS ExpenseFrequency,
	NULL as IsEssentialYN,
	NULL as EducationLevel,
	NULL as EducationStartDate,
	NULL as EducationEndDate,
	NULL as HealthIssueID,
	NULL as HealthIssue,
	NULL as DiagnosedYN,
	NULL as SelfReportedYN,
	NULL as SuspectedYN,
	NULL as HealthIssuesMedicationName,
	NULL as OtherMedications,
	--NULL as ClientWatchConcernID,
	--NULL as WatchConcern,
	NULL as ClientContributingFactorID,
	NULL AS ContributingFactor,
	NULL as ClientBehaviouralFactorID,
	NULL AS BehavioralFactor,
	NULL as Severity,
	lifeEvents.PeopleLifeEventID,
	lifeEvents.LifeEvent,
	NULL AS ClientDietID,
	NULL AS DietCatetory,
	NULL AS FoodType,
	NULL AS AvoidedDietYN,
	NULL as IntakeID,
	NULL as PreScreenPeriod,
	NULL as TotalScore,
	NULL AS ClientBarredPeriodID,
	NULL as Reason,
	NULL as Modules,
	NULL as ServiceRestricitonOrganizationName

FROM
	ClientBasics
	INNER JOIN lifeEvents ON ClientBasics.personID = lifeEvents.personID



UNION ALL

SELECT
	ClientBasics.*,
	NULL as ServiceID,
	'Diet' as ServiceType,
	Diets.CreatedDate as DateStart,
	NULL as DateEnd,
	NULL AS ReasonForService,
	NULL AS OrganizationName,
	NULL AS IncomeType,
	NULL as MonthlyAmount,
	NULL as ClientExpenseID,
	NULL AS ExpenseType,
	NULL AS ExpenseAmount,
	NULL AS ExpenseFrequency,
	NULL as IsEssentialYN,
	NULL as EducationLevel,
	NULL as EducationStartDate,
	NULL as EducationEndDate,
	NULL as HealthIssueID,
	NULL as HealthIssue,
	NULL as DiagnosedYN,
	NULL as SelfReportedYN,
	NULL as SuspectedYN,
	NULL as HealthIssuesMedicationName,
	NULL as OtherMedications,
	--NULL as ClientWatchConcernID,
	--NULL as WatchConcern,
	NULL as ClientContributingFactorID,
	NULL AS ContributingFactor,
	NULL as ClientBehaviouralFactorID,
	NULL AS BehavioralFactor,
	NULL as Severity,
	NULL as PeopleLifeEventID,
	NULL as LifeEvent,
	Diets.ClientDietID,
	Diets.DietCatetory,
	Diets.FoodType,
	Diets.AvoidedDietYN,
	NULL as IntakeID,
	NULL as PreScreenPeriod,
	NULL as TotalScore,
	NULL AS ClientBarredPeriodID,
	NULL as Reason,
	NULL as Modules,
	NULL as ServiceRestricitonOrganizationName


FROM
	ClientBasics
	INNER JOIN Diets ON ClientBasics.ClientID = Diets.ClientID



UNION ALL

SELECT
	ClientBasics.*,
	NULL as ServiceID,
	'VISPDAT' as ServiceType,
	VISPDATS.SPDAT_Date as StartDate,
	NULL as DateEnd,
	NULL AS ReasonForService,
	--NULL AS OrganizationName,
	VISPDATS.ServiceProvider as ORganizationName,
	NULL AS IncomeType,
	NULL as MonthlyAmount,
	NULL as ClientExpenseID,
	NULL AS ExpenseType,
	NULL AS ExpenseAmount,
	NULL AS ExpenseFrequency,
	NULL as IsEssentialYN,
	NULL as EducationLevel,
	NULL as EducationStartDate,
	NULL as EducationEndDate,
	NULL as HealthIssueID,
	NULL as HealthIssue,
	NULL as DiagnosedYN,
	NULL as SelfReportedYN,
	NULL as SuspectedYN,
	NULL as HealthIssuesMedicationName,
	NULL as OtherMedications,
	--NULL as ClientWatchConcernID,
	--NULL as WatchConcern,
	NULL as ClientContributingFactorID,
	NULL AS ContributingFactor,
	NULL as ClientBehaviouralFactorID,
	NULL AS BehavioralFactor,
	NULL as Severity,
	NULL as PeopleLifeEventID,
	NULL as LifeEvent,
	NULL AS ClientDietID,
	NULL AS DietCatetory,
	NULL AS FoodType,
	NULL AS AvoidedDietYN,
	VISPDATS.IntakeID,
	VISPDATS.PreScreenPeriod,
	VISPDATS.TotalScore,
	NULL AS ClientBarredPeriodID,
	NULL as Reason,
	NULL as Modules,
	NULL as ServiceRestricitonOrganizationName


FROM
	ClientBasics
	INNER JOIN VISPDATS ON ClientBasics.ClientID = VISPDATS.ClientID


UNION ALL

SELECT
	ClientBasics.*,
	NULL as ServiceID,
	'Service Restriction' as ServiceType,
	ServiceRestrictions.DateStart,
	ServiceRestrictions.DateEnd,
	NULL AS ReasonForService,
	NULL AS OrganizationName,
	NULL AS IncomeType,
	NULL as MonthlyAmount,
	NULL as ClientExpenseID,
	NULL AS ExpenseType,
	NULL AS ExpenseAmount,
	NULL AS ExpenseFrequency,
	NULL as IsEssentialYN,
	NULL as EducationLevel,
	NULL as EducationStartDate,
	NULL as EducationEndDate,
	NULL as HealthIssueID,
	NULL as HealthIssue,
	NULL as DiagnosedYN,
	NULL as SelfReportedYN,
	NULL as SuspectedYN,
	NULL as HealthIssuesMedicationName,
	NULL as OtherMedications,
	--NULL as ClientWatchConcernID,
	--NULL as WatchConcern,
	NULL as ClientContributingFactorID,
	NULL AS ContributingFactor,
	NULL as ClientBehaviouralFactorID,
	NULL AS BehavioralFactor,
	NULL as Severity,
	NULL as PeopleLifeEventID,
	NULL as LifeEvent,
	NULL AS ClientDietID,
	NULL AS DietCatetory,
	NULL AS FoodType,
	NULL AS AvoidedDietYN,
	NULL AS IntakeID,
	NULL AS PreScreenPeriod,
	NULL AS TotalScore,
	ServiceRestrictions.ClientBarredPeriodID,
	ServiceRestrictions.Reason,
	CAST(ServiceRestrictions.Modules as nvarchar) as Modules,
	ServiceRestrictions.OrganizationName as ServiceRestrictionOrganizationName


FROM
	ClientBasics
	INNER JOIN ServiceRestrictions ON ClientBasics.ClientID = ServiceRestrictions.ClientID


	
) t

WHERE 
	EXISTS (SELECT 1 FROM Services WHERE t.ClientID = Services.ClientID)

ORDER BY 
	ClientID,DateStart,DateEnd



