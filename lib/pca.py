import os
import matplotlib.pyplot as plt
import pandas as pd
from joblib.numpy_pickle_utils import xrange
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LinearSegmentedColormap
import colorsys
import numpy as np


class Pca:
    def __init__(self, csv, n_cluster=0, autocompute=True, verbose=0):
        # --- Preparazione ---
        self.verbose = verbose

        self.targets = []
        self.url = csv
        if n_cluster == 0:
            try:
                self.n_cluster = int(self.url[self.url.rfind('_') + 1:-4])  # ricavo numero di cluster da nome del csv
            except ValueError:
                self.n_cluster = int(input("Insert number of clusters: "))
        else:
            self.n_cluster = n_cluster

        if self.n_cluster < 0:
            print("n_cluster must be an integer > 0")
            return

        self.new_cmap = self.rand_cmap(self.n_cluster, 'bright', True, False, verbose)

        # self.url = "CSV/" + csv  # Prende il csv dalla cartella CSV.

        # Provo ad aprire il file, se non riesco esco con codice -1, altrimenti lo chiudo.
        try:
            f = open(self.url)
        except IOError:
            print("Non è possibile aprire " + self.url)
            exit(-1)
        f.close()

        # Prepara dinamicamante l'array ['cluster0','cluster1',...'clusterN']
        for n in range(self.n_cluster):
            self.targets.append('cluster' + str(n))
        if (self.verbose >= 1):
            print(self.targets)

        if autocompute:
            self.compute()

    def compute(self):
        # --- Computazione PCA ---
        # Questa è la struttura del file '.csv'.
        self.df = pd.read_csv(self.url,
                              names=['DZCOM', 'DENSPOP', 'AGMAX_50', 'IDR_POPP3', 'IDR_POPP2', 'IDR_POPP1',
                                     'IDR_AREAP1',
                                     'IDR_AREAP2', 'IDR_AREAP3', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E19',
                                     'E20',
                                     'E30', 'E31', 'Cluster'])

        features = ['DENSPOP', 'AGMAX_50', 'IDR_POPP3', 'IDR_POPP2', 'IDR_POPP1', 'IDR_AREAP1', 'IDR_AREAP2',
                    'IDR_AREAP3', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E19', 'E20', 'E30',
                    'E31']  # Identifica gli attributi

        x = self.df.loc[:, features].values  # Raccoglie solo esempi con attributi di features
        x = StandardScaler().fit_transform(x)  # Normalizza i dati:
        # fit_transform calcola prima la media e la std (fit) e poi le porta rispettivamente a 0 e a 1
        # trasformando i dati adeguatamente (transform).

        pca = PCA(n_components=3)  # PCA con tre componenti
        self.principalComponents = pca.fit_transform(x)  # raccoglie le componenti principali e trasforma i dati
        if (self.verbose >= 2):
            print(self.principalComponents)
        principalDf = pd.DataFrame(data=self.principalComponents,
                                   columns=['principal component 1', 'principal component 2', 'principal component 3'])
        # costruisce il dataset formato dalle componenti principali
        self.finalDf = pd.concat([principalDf, self.df['Cluster'], self.df['DZCOM']], axis=1)
        # costruisco dataset completo di componenti principali, cluster e denominazione comune

    def plot(self, save=False, names=False):
        # --- Plot dei risultati ---
        # Prepara il nome per grafico e immagine
        self.img_name = self.url[:-4]
        self.img_name = self.img_name.replace('/', '_')
        self.img_name = self.img_name[4:self.img_name.rfind('_')]
        if names:
            self.img_name = self.img_name + "_names_2D"
        else:
            self.img_name = self.img_name + "_clean_2D"

        self.img_ext = ".png"

        self.img_title = self.img_name.replace('_', ' ')

        # Plot
        self.fig = plt.figure(figsize=(25, 13))  # creo dimensioni figura
        ax = self.fig.add_subplot(1, 1, 1)  # creo finestra di dimensione 1x1, posizionato nel 1 quadrante
        ax.set_xlabel('Principal Component 1', fontsize=15)  # label asse x
        ax.set_ylabel('Principal Component 2', fontsize=15)  # label asse y
        ax.set_title(self.img_title, fontsize=20)  # label titolo

        for target in self.targets:
            indicesToKeep = self.finalDf['Cluster'] == target
            ax.scatter(self.finalDf.loc[indicesToKeep, 'principal component 1']
                       , self.finalDf.loc[indicesToKeep, 'principal component 2']
                       , cmap=self.new_cmap
                       , s=10)  # creo insieme di punti
        ax.legend(self.targets)  # inserisce legenda
        ax.grid()  # inserisce griglia

        if names:
            for i, txt in enumerate(self.df['DZCOM']):
                ax.annotate(txt, (
                    self.principalComponents[i][0], self.principalComponents[i][1]))  # aggiunta etichetta a ogni punto
        if (save):
            self.saving()

    def plot3(self, save=False, names=False):
        # --- Plot 3D dei risultati ---
        # Prepara il nome per grafico e immagine

        self.img_name = self.url[:-4]
        self.img_name = self.img_name.replace('/', '_')
        self.img_name = self.img_name[4:self.img_name.rfind('_')]
        if names:
            self.img_name = self.img_name + "_names_3D"
        else:
            self.img_name = self.img_name + "_clean_3D"

        self.img_ext = ".png"

        self.img_title = self.img_name.replace('_', ' ')

        # Plot 3D
        self.fig = plt.figure(figsize=(25, 13))
        axes = plt.axes(projection="3d")
        axes.set_title(self.img_title, fontsize=20)
        axes.set_xlabel("Principal Component 1")
        axes.set_ylabel("Principal Component 2")
        axes.set_zlabel("Principal Component 3")

        for i, target in zip(enumerate(self.targets), self.targets):
            indicesToKeep = self.finalDf['Cluster'] == target
            axes.scatter3D(self.finalDf.loc[indicesToKeep, 'principal component 1']
                           , self.finalDf.loc[indicesToKeep, 'principal component 2']
                           , self.finalDf.loc[indicesToKeep, 'principal component 3']
                           , cmap=self.new_cmap, s=10
                           )
        axes.legend(self.targets)
        axes.grid()

        if names:
            for i, txt in enumerate(self.df['DZCOM']):
                # axes.annotate(txt, (self.principalComponents[i][0], self.principalComponents[i][1], self.principalComponents[i][2])) non va
                axes.text(self.principalComponents[i][0],
                          self.principalComponents[i][1],
                          self.principalComponents[i][2],
                          txt, size=8, zorder=1, color='k')
        if (save):
            self.saving()

    def show(self):
        plt.show()

    def saving(self):
        path = "img_pca"
        try:
            os.mkdir(path)
        except FileExistsError:
            if (self.verbose >= 1):
                print("%s exists yet" % path)
        self.fig.savefig(path + "/" + self.img_name + self.img_ext)

    def rand_cmap(self, nlabels, type, first_color_black, last_color_black, verbose):
        # Questo metodo serve a generare una mappa di colori dinamica (della stessa grandezza di nlabels).
        # I colori sono randomici, ma per avere contiguità tra i colori dei vari plot, è stato deciso di tenere un
        # seed prefissato a zero.
        # https://github.com/delestro/rand_cmap
        """
        Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
        :param nlabels: Number of labels (size of colormap)
        :param type: 'bright' for strong colors, 'soft' for pastel colors
        :param first_color_black: Option to use first color as black, True or False
        :param last_color_black: Option to use last color as black, True or False
        :param verbose: Prints the number of labels and shows the colormap. Zero or greater
        :return: colormap for matplotlib
        """
        if type not in ('bright', 'soft'):
            print('Please choose "bright" or "soft" for type')
            return

        if verbose >= 2:
            print('Number of labels: ' + str(nlabels))

        np.random.seed(0)

        # Generate color map for bright colors, based on hsv
        if type == 'bright':
            randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                              np.random.uniform(low=0.2, high=1),
                              np.random.uniform(low=0.9, high=1)) for i in xrange(nlabels)]

            # Convert HSV list to RGB
            randRGBcolors = []
            for HSVcolor in randHSVcolors:
                randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

            if first_color_black:
                randRGBcolors[0] = [0, 0, 0]

            if last_color_black:
                randRGBcolors[-1] = [0, 0, 0]

            random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

        # Generate soft pastel colors, by limiting the RGB spectrum
        if type == 'soft':
            low = 0.6
            high = 0.95
            randRGBcolors = [(np.random.uniform(low=low, high=high),
                              np.random.uniform(low=low, high=high),
                              np.random.uniform(low=low, high=high)) for i in xrange(nlabels)]

            if first_color_black:
                randRGBcolors[0] = [0, 0, 0]

            if last_color_black:
                randRGBcolors[-1] = [0, 0, 0]
            random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

        # Display colorbar
        if verbose >= 2:
            from matplotlib import colors, colorbar
            from matplotlib import pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

            bounds = np.linspace(0, nlabels, nlabels + 1)
            norm = colors.BoundaryNorm(bounds, nlabels)

            cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                       boundaries=bounds, format='%1i', orientation=u'horizontal')

        return random_colormap
