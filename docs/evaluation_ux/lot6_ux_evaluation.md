# Evaluation UX lot 6 - optimal_tf dashboard

Date: 2026-07-02
URL testee: `http://localhost:8502/`

## 1. Verdict synthese

### Non-regression produit

Verdict: PASS

Le produit conserve bien l'architecture cible observee sur les lots precedents:

- `Workspace`
- `Run`
- `Compare`
- `Search`
- `Guide`

Les 12 vues attendues sont accessibles et se chargent sans erreur bloquante:

- `Workspace / Config editor`
- `Run / Allocation`
- `Run / Evaluation`
- `Run / Inspection snapshot`
- `Compare / Compare`
- `Compare / Vary strategy`
- `Compare / Vary cleaning`
- `Compare / Vary window`
- `Compare / Vary frequency`
- `Search / Strategy testbed`
- `Search / Hyperparameter tuning`
- `Guide / Strategy guide`

Chaque vue expose un CTA principal coherent avec sa promesse:

- `Run allocation`
- `Run evaluation`
- `Run inspection snapshot`
- `Run compare`
- `Run vary strategy`
- `Run vary cleaning`
- `Run vary window`
- `Run vary frequency`
- `Run strategy testbed`
- `Run hyperparameter tuning`

### Evaluation UX

Verdict: PASS avec reserves

Le lot 6 franchit un vrai cap en coherence produit. La structure globale est maintenant lisible, les services sont mieux differencies, et les textes d'aide expliquent plus clairement le role de chaque vue. En revanche, l'experience reste encore inegale sur trois points: densite du `Workspace`, latence de navigation entre services, et ecart de maturite entre les vues les plus pedagogiques et les vues encore tres "formulaires".

## 2. Resultats de non-regression

### Parcours verifies

- Bascule entre les 5 modes fonctionnelle.
- Changement de service fonctionnel dans chaque mode.
- Rendu correct des titres de page et des CTA principaux.
- Presence d'un descriptif service dans toutes les vues.
- Presence d'alertes pedagogiques dans `Run`, `Compare`, `Search` et `Guide`.

### Temps de transition observes

Temps observes pour charger la vue apres changement de service:

- `Workspace / Config editor`: ~3010 ms
- `Run / Allocation`: ~3607 ms
- `Run / Evaluation`: ~3304 ms
- `Run / Inspection snapshot`: ~3122 ms
- `Compare / Compare`: ~4021 ms
- `Compare / Vary strategy`: ~4010 ms
- `Compare / Vary cleaning`: ~3337 ms
- `Compare / Vary window`: ~3221 ms
- `Compare / Vary frequency`: ~3859 ms
- `Search / Strategy testbed`: ~4009 ms
- `Search / Hyperparameter tuning`: ~2708 ms
- `Guide / Strategy guide`: ~4004 ms

Conclusion produit: pas de regression structurelle ou de service manquant constatee, mais la reactivite reste percue comme moyenne et encore heterogene.

### Console / signaux techniques visibles

Warning recurrent observe pendant la navigation:

- ``preventOverflow` modifier is required by `hide` modifier in order to work, be sure to include it before `hide`!`

Ce warning n'empeche pas l'usage, mais il signale un probleme d'integration UI a corriger.

## 3. Evaluation de coherence UX

### Ce qui est mieux qu'avant

- La segmentation par modes `Workspace / Run / Compare / Search / Guide` est maintenant solide et comprensible.
- Les noms de services sont beaucoup plus explicites et alignes avec les intentions produit.
- Les alertes contextuelles jouent un vrai role de guidage et rapprochent l'ensemble de la qualite percue de `Strategy testbed`.
- Les CTA finaux suivent une convention stable: `Run ...`.
- `Compare` est devenu un bloc produit coherent, avec des variantes facilement comprenables.

### Ce qui reste incoherent

- `Workspace / Config editor` reste beaucoup plus dense et plus "technique brute" que le reste du produit.
- `Guide / Strategy guide` est utile comme orientation, mais parait encore trop leger face au niveau d'explication present dans `Run` et `Search`.
- Certaines vues montrent un bon niveau d'accompagnement pedagogique, mais d'autres restent surtout des formulaires avec peu de hierarchie visuelle.
- Les controles de configuration sont souvent similaires d'un service a l'autre, mais leur mise en scene ne donne pas toujours l'impression d'un socle commun clairement structure.

### Reference UX actuelle

`Search / Strategy testbed` reste la UX la plus aboutie.

Pourquoi:

- intention tres claire
- bon niveau de guidage
- bon decoupage entre contexte, parametrage et action
- sensation d'espace de travail specialise plutot que simple formulaire

Le lot 6 rapproche `Run` et `Compare` de cette logique, mais `Workspace` et `Guide` sont encore en retrait.

## 4. Evaluation de la facilite d'utilisation

### Points forts

- Le choix du mode puis du service est simple a comprendre.
- Chaque service annonce mieux son usage des le haut de page.
- La relation entre la vue et son CTA principal est claire.
- Les alertes aident a choisir entre services proches comme `Allocation`, `Evaluation` et `Inspection snapshot`.

### Frictions principales

- Le panneau lateral reste lourd: il concentre navigation, contexte, description et configuration rapide dans un espace etroit.
- Les temps de bascule entre services cassent un peu l'impression de fluidite.
- Les formulaires restent longs, avec une charge cognitive importante avant de pouvoir lancer une action.
- `Workspace` expose encore beaucoup de detail d'un coup, sans vrai parcours de simplification.
- `Guide` oriente, mais n'aide pas encore assez au choix concret du "bon prochain service".

## 5. Points prioritaires a corriger

### Priorite 1 - unifier le haut de page de tous les services

Mettre partout la meme structure:

- promesse du service
- quand l'utiliser
- ce qu'il produit
- CTA principal

`Run`, `Compare` et `Search` sont proches de cette cible. `Workspace` et `Guide` doivent etre remis au meme niveau.

### Priorite 2 - alleger le `Workspace`

Le transformer d'une page "config brute" vers une page "socle produit":

- contexte courant
- parametres essentiels
- section avancee repliee
- sorties et chemins secondaires moins dominants

### Priorite 3 - rendre `Guide` plus operationnel

Le `Guide` devrait aider a choisir:

- quelle famille de strategie explorer
- quel service lancer ensuite
- quel niveau d'exploration choisir (`Run`, `Compare`, `Search`)

Aujourd'hui il informe; demain il devrait orienter.

### Priorite 4 - lisser les temps de transition

Objectif UX recommande:

- viser une sensation de bascule < 1 s pour les changements de service simples
- afficher un feedback de chargement plus explicite quand la vue prend plus de temps

## 6. Conclusion

Le lot 6 est non regressif sur le produit et marque une progression nette sur la coherence UX. L'architecture de l'offre est maintenant stable et lisible, `Run` et `Compare` sont plus pedagogiques, et `Search / Strategy testbed` reste la reference vers laquelle le reste converge.

Le chantier UX n'est cependant pas termine: le prochain gain majeur viendra moins d'une nouvelle architecture que d'un polissage transversal. Les priorites les plus rentables sont l'allegement du `Workspace`, la montee en puissance du `Guide`, et une harmonisation encore plus stricte de la structure d'entree de chaque service.
