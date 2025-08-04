param (
    [int]$id_client
)

# Cambia directory in src (dove si trova client.py)
Set-Location ".\src"

# Genera numero casuale tra 0 e 1
$numero_casuale = Get-Random
$numero_casuale = [math]::Round(($numero_casuale / [int]::MaxValue), 4)

Write-Host "Numero casuale generato: $numero_casuale"

# Definisce i percorsi ai dataset
$dataset_dir1 = "..\data\brain_tumor_dataset"
$dataset_dir2 = "..\data\brain_tumor_dataset2"

# Scegli il dataset in base al numero casuale
if ($numero_casuale -ge 0.5) {
    Write-Host "Il numero casuale è maggiore o uguale a 0.5"
    python client.py $dataset_dir1 $id_client
} else {
    Write-Host "Il numero casuale è minore di 0.5"
    python client.py $dataset_dir2 $id_client
}
