##########################################################################################################
##
##   Script to export HIFIS Data for consumption by Azure Machine Learning Studio
##
##
##   This script will run the SQLPS commands to generate two CSV files for upload to azure blob storage
##   Source SQL Scripts are found in the src\data\queries folder
##   csv files are saved to data\raw\ folder.
##
##   CSV files are uploaded to Azure Blob storage for consumption by Azure Machine Learning Studio.
##
##
#########################################################################################################

Import-Module SQLPS -DisableNameChecking

$VerbosePreference = "SilentlyContinue"

## Database Instance Name. Place it in the following line.
$instanceName = "[INSTANCE NAME\DATABASE NAME]"

# SQL Scripts
$clientScript = "src\data\queries\client_export.sql"
$spdatScript = "src\data\queries\SPDAT_export.sql"

# Output Files
$clientOutputFile = "data\raw\HIFIS_Clients.csv"
$spdatOutputFile = "data\raw\SPDATS.json"

# export client data to csv file
Invoke-Sqlcmd -InputFile $clientScript -ServerInstance $instanceName -QueryTimeout 0 | Export-Csv -Path $clientOutputFile -NoTypeInformation


# export spdat data to JSON file
Invoke-Sqlcmd -inputfile $spdatScript -ServerInstance $instanceName | Select-Object -ExpandProperty JSON_F52E2B61-18A1-11d1-B105-00805F49916B | Out-File $spdatOutputFile

#Upload to Azure BLOB Storage



If ((Test-Path -Path $clientOutputFile) -and (Test-Path -Path $spdatOutputFile))

{

    # connect to Azure using VM managaed Identity
    # Managed identities for Azure resources provides Azure services with an automatically managed identity in Azure Active Directory. 
    # You can use this identity to authenticate to any service that supports Azure AD authentication, without having credentials in your code.
    # https://docs.microsoft.com/en-us/azure/active-directory/managed-identities-azure-resources/qs-configure-portal-windows-vm

    $response = Invoke-WebRequest -Uri 'http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https%3A%2F%2Fmanagement.azure.com%2F' -Method GET -Headers @{Metadata="true"} -UseBasicParsing

    # Extract "Content" element from the response
    $content = $response.Content | ConvertFrom-Json

    # Extract the access token from the response
    $ArmToken = $content.access_token

    # Get Storage Account Keys from Azure Resource Manager
    $keysResponse = Invoke-WebRequest -Uri https://management.azure.com/subscriptions/[SUBSCTRIPTION GUID]/resourceGroups/[RESOURCE GROUP NAME]/providers/Microsoft.Storage/storageAccounts/[STORAGE ACCOUNT NAME]/listKeys/?api-version=2016-12-01 -Method POST -Headers @{Authorization="Bearer $ARMToken"} -UseBasicParsing

    # Get "Content" element from key response
    $keysContent = $keysResponse.Content | ConvertFrom-Json

    # Get key from keysContent
    $key = $keysContent.keys[0].value
        

    # Get Azure Storage Context
    $ctx = New-AzureStorageContext -StorageAccountName [STORAGE ACCOUNT NAME] -StorageAccountKey $key

    # Upload HIFIS_Clients.csv File
    Set-AzureStorageBlobContent -File $clientOutputFile -Container [CONTAINER NAME] -blob [BLOB FOLDER]/HIFIS_Clients.csv -Context $ctx -Force

    # Upload HIFIS_SPDATS.csv File
    Set-AzureStorageBlobContent -File $spdatOutputFile -Container [CONTAINER NAME] -blob [BLOB FOLDER]/SPDATS.json -Context $ctx -Force

}



EXIT