# Omezení experimentů (statický denní model)

Tento dokument shrnuje interpretační limity experimentů **exp02 / exp04 / exp06 / exp08**,
které pracují s **jednodenním statickým přiřazením** (BFW na průměrné denní matice) a
ne s hodinovým dynamickým modelem.

## 1. Lokální uzávěry a ΔVHT (exp02, exp04, exp08)

- **Malý relativní dopad**: Uzávěry na málo hranách mění jen malou část celkového VHT
  v celé síti; ΔVHT v řádu **zlomků procenta** je očekávatelný, ne chyba výpočtu.
- **Částečné uzávěry (lane reduction)**: Dříve byl v `apply_scenario_to_graph` chybný
  spodní limit `lanes_remaining >= 1`, takže se různé „závažnosti“ scénáře mohly
  chovat stejně. Po opravě závisí výsledek na skutečném počtu jízdních pruhů na hraně.
- **Citlivost na RGAP / poptávku** (exp08): Globální škálování OD nebo přísnější RGAP
  mění rovnováhu na celé síti — interpretujte jako **systémovou** citlivost, ne
  lokální dopad jedné uzávěry.

## 2. Korelace model × Waze (exp03, exp06, exp09)

- Denní **V/C** z rovnovážného přiřazení není přímým ekvivalentem hodinové frekvence
  zácp ve Waze. **Slabá Spearmanova korelace** tedy nemusí znamenat „špatný model“,
  ale nesoulad agregace času a definice zácpy.
- **exp06 (precision / recall)**: Globální pravidlo „všechny linky s V/C > 1“ vede k
  obrovskému počtu falešně pozitivních oproti malé množině jam linků u konkrétní
  události. Skript proto používá **lokální buffer** kolem uzávěry, **adaptivní
  práh V/C** (percentil v bufferu) a **prostorové párování** jam linků s modelem
  (geometrie linek blíž než ~55 m), protože mapování Waze segment → síťový link
  často nesedí s přesným archem s nejvyšším V/C.

## 3. quality_score (exp10)

- Pokud je v datech málo aktivních uzávěr s geografickými souřadnicemi, křivky
  citlivosti mohou být i nadále ploché. Experiment **sdružuje více uzávěr** do jedné
  sítě scénáře podle prahu `quality_score`, aby šlo sledovat monotónní nárůst počtu
  zasažených hran.

## Doporučení pro text diplomové práce

1. Vždy uvést **časovou agregaci** (denní vs hodinový) a zdroj externích dat (CSD,
   Waze, uzávěry z DB).
2. U scénářů uzávěrek zdůraznit **topologii** (kolik hran, jak daleko od centra
   poptávky) a že statický model **nepopisuje fronty v čase**.
3. U validace vůči Waze uvést **geografické okno** (buffer) a práh V/C, pokud se
   používají proxy metriky.
