Lo script _**start_client.sh**_ è uno script di shell progettato per avviare il client che interagisce con il server nel progetto di classificazione delle immagini di tumori cerebrali. Utilizza un meccanismo di selezione casuale per scegliere quale dataset utilizzare durante l'esecuzione.

**client.py:** rappresenta il client nel FL.

**Model.py:** definisce l'architettura del modello di deep learning per la classificazione delle immagini. Implementa una rete neurale convoluzionale (CNN).

**server.py:** rappresenta il server per il FL.

**_server.sh:_** avvia il server eseguendo server.py, rendendolo pronto a ricevere connessioni dal client.

**dataset.py:** gestisce il caricamento e la pre-elaborazione dei dataset di immagini.
