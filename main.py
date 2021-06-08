import os
from pat import Pat

print("PCA And Tree.\nAlessandro Gilli, Luana Mantovan - 2021")

n_cluster=0         #Default = 0, con 0 prende in automatico il numero di cluster dal nome

names=False         #Default = False
plot_save=True     #Default = False

depth=4             #Default = None
tree_save=True      #Default = False
desc_save=True      #Default = False
text_save=True      #Default = False
rules_save=True     #Default = False

mmenu = """
Funzioni PCA:
[1] Plot
[2] Plot 3D
Funzioni Tree:
[3] Plot
[4] Description
[5] Text
[6] Rules
Sistema:
[e] Uscita
[c] Carica altro .csv
"""
cmd="c"

while cmd != "e":
    if(cmd == "c"):
        print("Scegliere un dataset da cartella CSV:")
        dirs = os.listdir('CSV/')
        for i,dss in enumerate(dirs):
            print(f"[{i}] {dss}")
        cmd = input("Scelta: ")
        csv = "CSV/" + dirs[int(cmd)]
        p = Pat(csv,depth=depth,n_cluster=n_cluster)

    elif cmd == "1":
        p.pca.plot(names=names,save=plot_save)
        p.pca.show()
    elif cmd == "2":
        p.pca.plot3(names=names,save=plot_save)
        p.pca.show()
    elif cmd == "3":
        p.tree.plot(save=tree_save)
        p.tree.show()
    elif cmd == "4":
        p.tree.description(save=desc_save)
        input("Premere invio per continuare...")
    elif cmd == "5":
        p.tree.text(save=text_save)
        input("Premere invio per continuare...")
    elif cmd == "6":
        p.tree.rules(save=rules_save)
        input("Premere invio per continuare...")
    else:
        print("Inserire il numero associato al comando")

    print(mmenu)
    cmd = input("Scelta: ")