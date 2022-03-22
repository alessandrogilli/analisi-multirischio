# analisi-multirischio
Progetto python per visualizzare classi di rischio sismico e idrogeologico da dataset Istat opportunamente modificati. La guida si suddivide in step: dalla creazione dei
file .csv creati con Weka (o con un altro qualsiasi strumento per il clustering) alla visualizzazione dei risultati.

## Creazione file .csv
Questa operazione si compone delle seguenti fasi:
* Riduzione del dataset Istat selezionando solo i seguenti attributi:
  * DZCOM
  * DENSPOP
  * AGMAX_50
  * IDR_POPP3
  * IDR_POPP2
  * IDR_POPP1
  * IDR_AREAP1
  * IDR_AREAP2
  * IDR_AREAP3
  * E5
  * E6
  * E7
  * E8
  * E9
  * E10
  * E11
  * E19
  * E20
  * E30
  * E31;
* Normalizzazione dei dati;
* Calcolo dei cluster tramite algoritmo k-means o Expectation-Maximization (con qualsiasi applicativo, l'importante è mantenere coerente il layout dei file in output, vedi punti successivi);
* Rinomina del file .csv secondo il seguente schema:
    *numVersione_Algoritmo_nCluster.csv*
    
  Dove:
    * *numVersione* è un numero progressivo che consiste di visualizzare la versione creata
    * *Algoritmo* indica se è stato utilizzato k-means (KMEANS) o EM (EM)
    * *nCluster* indica il numero di cluster risultati dall'algoritmo, necessario per i passaggi successivi;
* I file .csv devono essere contenuti nella cartella CSV.

## Architettura 
Il progetto è costituito da un file principale, main, che consente di visualizzare un menu per compiere determinate operazioni, segnalate come *Funzioni PCA* ([1] e [2]), *Funzioni Tree* ([3], [4], [5] e [6]), *Sistema* ([e] e [c]):

1. *[1] Plot*: consente di stampare in 2D il risultato della PCA, quindi di visualizzare i cluster in un grafico in due dimensioni;
2. *[2] Plot 3D*: consente di ottenere lo stesso risultato di Plot ma in un grafico 3D;
3. *[3] Plot*: permette di creare un immagine .jpg che rappresenta l'albero di decisione relativo al risultato ottenuto con la PCA;
4. *[4] Description*: fornisce una descrizione schematica dell'albero di decisione. In altre parole, l'albero di decisione viene trascritto in un file di testo;
5. *[5] Text*: In maniera più verbosa, viene fornito sotto forma di file di testo l'albero di decisione, specificando quali strade percorrere e quali nodi seguire;
6. *[6] Rules*: Stampa in forma testuale le regole di decisione generate tramite la creazione dell'albero. Il file specifica anche per ogni nodo qual è la probabilità che un esempio possa essere correttamente posizionato in quel determinato nodo;
7. *[e]  Uscita*: chiusura dell'esecuzione;
8. *[c] Carica altro .csv*: consente di scegliere un altro file .csv per visualizzare i risultati. I nomi dei file .csv verranno mostrati a video in un elenco numerato e sarà possibile scegliere il numero desiderato da riga di comando.

All'interno del file *main* si trovano anche delle variabili booleane e numeriche che possono essere modificate a piacimento per customizzare la visualizzazione dei risultati:
* *names*: se a True, stampa all'interno del grafico in 2D la denominazione del comune di ogni punto analizzato - di default è a False;
* *names3*: se a True, stampa all'interno del grafico in 3D la denominazione del comune di ogni punto analizzato - di default è a False;
* *plot_save*: se impostato a True, salva il grafico 2D in un file .jpeg - di default è a False;
* *plot_save*: se impostato a True, salva il grafico 3D in un file .jpeg - di default è a False;
* *depth*: variabile numerica per definire l'altezza dell'albero di decisione. Impostata di default a None in quanto, in questa maniera, genererebbe in maniera automatica l'altezza dell'albero classificando correttamente tutti gli esempi dati;
* *tree_save*: se impostato a True, salva l'immagine dell'albero di decisione - di default è a False;
* *desc_save*: se impostato a True, salva l'esecuzione di quanto eseguito durante la scelta n.4, Description - di default è a False;
* *text_save*: se impostato a True, salva l'esecuzione di quanto eseguito durante la scelta n.5, Text - di default è a False;
* *rules_save*: se impostato a True, salva l'esecuzione di quanto eseguito durante la scelta n.6, Rules - di default è a False.

## Creazione ambiente virtuale ed esecuzione del programma
Potrebbe essere opportuno impostare un ambiente virtuale nel quale installare tutte le librerie necessarie per la corretta esecuzione del programma.

### Creazione ambiente virtuale
Sia per utenti Windows che Mac/Linux fare riferimento alla guida ufficiale:
https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/

### Installazione librerie necessarie
Installare le librerie necessarie all'interprete (assicurarsi di avere attivato l'ambiente virtuale):
```python
python -m pip install matplotlib pandas sklearn numpy 
```
### Eseguire il programma
Per eseguire il programma, assicurarsi di avere attivato l'ambiente virtuale e di essere dentro la directory del progetto, poi eseguire:
```python
python main.py
```
Il programma visualizzerà come primo menu la lista dei file .csv contenuti all'interno della cartella CSV e successivamente il menu con le varie opzioni per visualizzare i risultati.

## Autori
- [Alessandro Gilli](mailto:alessandro.gilli@edu.unife.it)
- [Luana Mantovan](mailto:luana.mantovan@edu.unife.it)
