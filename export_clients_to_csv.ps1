##########################################################################################################
##
##   Script to export HIFIS Data for consumption by the preprocessing script.
##
##
##   This script will run the SQLCMD utility to generate two CSV files
##   Source SQL Script is found at src\data\ClientExport.sql
##   csv file is saved to data\raw\HIFIS_Clients.csv
##
#########################################################################################################

Import-Module SQLPS -DisableNameChecking

$VerbosePreference = "SilentlyContinue"

## Database Instance Name. Place it in the following line.
$instanceName =  [Instance Name goes here] 

# SQL Scripts
$clientScript = "src\data\client_export.sql" # Path to SQL Script


# Output Files
$clientOutputFile = "data\raw\HIFIS_Clients.csv"


# export client data to csv file
Invoke-Sqlcmd -InputFile $clientScript -ServerInstance $instanceName  | Export-Csv -Path $clientOutputFile -NoTypeInformation





EXIT