## English

### Project Description
The goal of this project is to build a model that can accurately predict whether a client is interested in buying car insurance. This is a binary classification problem based on customer demographics and vehicle information.

### The Challenge: Class Imbalance
A key characteristic of the dataset is a significant class imbalance. Only about 12% of customers show interest in the insurance policy, while the remaining 88% do not. The primary challenge is to build models that can effectively identify the minority class (interested customers) without being biased by the majority class.

### Methodology
Both the Python and R reports follow a similar analytical process:
1.  **Exploratory Data Analysis (EDA):** Initial inspection of data distributions, correlations, and outliers.
2.  **Data Pre-processing:** Conversion of categorical features (like `Gender`, `Vehicle_Age`) into numerical formats suitable for modeling.
3.  **Handling Class Imbalance:** Application of techniques to create balanced training datasets, including:
    *   Random Undersampling
    *   SMOTE (Synthetic Minority Over-sampling Technique)
    *   A hybrid approach combining SMOTE and Undersampling
4.  **Model Building & Evaluation:** Training and evaluation of several classification models, assessing their performance using metrics like accuracy, sensitivity, specificity, and ROC curves.

### Models Explored
*   Decision Trees
*   Naive Bayes
*   Support Vector Machines (SVM), with and without class weighting
*   XGBoost

### Key Findings
*   Addressing the class imbalance is crucial for building effective models. Models trained on the original, imbalanced data perform poorly in identifying interested customers.
*   Balancing techniques like Random Undersampling and hybrid methods significantly improve model performance, especially for the minority class.
*   The **XGBoost** model, particularly when trained on a balanced dataset or using class weights, provides the best combination of predictive performance and fast execution time.
*   SVM also shows strong performance but with significantly longer training times, making it less practical for this use case.

---

## Italiano

### Descrizione del Progetto
L'obiettivo di questo progetto è costruire un modello in grado di prevedere con accuratezza se un cliente è interessato all'acquisto di una polizza di assicurazione auto. Si tratta di un problema di classificazione binaria basato sui dati anagrafici dei clienti e sulle informazioni relative al loro veicolo.

### La Sfida: Sbilanciamento delle Classi
Una caratteristica chiave del dataset è un forte sbilanciamento delle classi. Solo il 12% circa dei clienti mostra interesse per la polizza, mentre il restante 88% non è interessato. La sfida principale è costruire modelli che possano identificare efficacemente la classe di minoranza (clienti interessati) senza essere influenzati dalla classe di maggioranza.

### Metodologia
Entrambi i report, in Python e in R, seguono un processo analitico simile:
1.  **Analisi Esplorativa dei Dati (EDA):** Ispezione iniziale delle distribuzioni dei dati, delle correlazioni e degli outlier.
2.  **Pre-processing dei Dati:** Conversione delle feature categoriche (come `Gender`, `Vehicle_Age`) in formati numerici adatti alla modellazione.
3.  **Gestione dello Sbilanciamento delle Classi:** Applicazione di tecniche per creare training set bilanciati, tra cui:
    *   Random Undersampling (sottocampionamento casuale)
    *   SMOTE (Synthetic Minority Over-sampling Technique)
    *   Un approccio ibrido che combina SMOTE e Undersampling
4.  **Creazione e Valutazione dei Modelli:** Addestramento e valutazione di diversi modelli di classificazione, misurandone le performance con metriche quali accuratezza, sensibilità, specificità e curve ROC.

### Modelli Analizzati
*   Alberi Decisionali
*   Naive Bayes
*   Support Vector Machines (SVM), con e senza pesatura delle classi
*   XGBoost

### Risultati Chiave
*   Affrontare lo sbilanciamento delle classi è fondamentale per costruire modelli efficaci. I modelli addestrati sul dataset originale e sbilanciato hanno scarse performance nell'identificare i clienti interessati.
*   Le tecniche di bilanciamento, come il Random Undersampling e i metodi ibridi, migliorano significativamente le prestazioni del modello, specialmente per la classe di minoranza.
*   Il modello **XGBoost**, in particolare quando addestrato su un dataset bilanciato o utilizzando pesi per le classi, fornisce la migliore combinazione di performance predittiva e velocità di esecuzione.
*   Anche le SVM mostrano ottime performance, ma con tempi di addestramento significativamente più lunghi, che le rendono meno pratiche per questo caso d'uso.
