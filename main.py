# IMPORTAZIONE DELLE LIBRERIE
import pandas as pd  # PER STRUTTURE DATI E MANIPOLAZIONE (DF/SERIES)
import seaborn as sns  # PER PLOT
import matplotlib.pyplot as plt  # PER PLOT

# CREAZIONE DEL DATASET E VISUALIZZAZIONE DI DIMENSIONI
# VISUALIZZAZIONI TIPI DI DATO DELLE COLONNE
df = pd.read_csv('insurance.csv')
df.head()
# Alcune feature necessitano di essere manipolate prima di essere utilizzate
# Ad esempio la colonna id viene tolta in quanto irrilevante ai fini dell'analisi
df.describe()

# Mostro i valori unici presenti in ogni colonna
df = df.drop(columns='id')

for column in df.columns:
    print(f"{column}: ")
    print("")
    print(df[column].unique())
    print("")

df.describe(include='O')

#BOXPLOT DI AGE, ANNUAL_PREMIUM E VINTAGE
plt.subplot(1, 3, 1)
plt.boxplot(df[['Age']], flierprops={'marker': 'o', 'markersize': 2})
plt.subplot(1, 3, 2)
plt.boxplot(df[['Annual_Premium']], flierprops={'markersize': 2})
plt.subplot(1, 3, 3)
plt.boxplot(df[['Vintage']], flierprops={'markersize': 2})
plt.show()

