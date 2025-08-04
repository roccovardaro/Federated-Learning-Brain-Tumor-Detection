param (
    [string]$dataset_dir,
    [int]$NUM_CLIENTS
)

# Cambia directory nella cartella 'src'
cd src
# Avvia i client
for ($i = 1; $i -le $NUM_CLIENTS; $i++) {
    Write-Host "Istanza del client $i avviata"
    Start-Process -NoNewWindow -FilePath "python" -ArgumentList "client.py", "$dataset_dir", "$i"
}
