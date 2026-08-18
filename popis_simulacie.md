# Ako funguje táto simulácia — vysvetlenie od nuly

> Tento dokument vysvetľuje, čo kód v tomto repozitári robí, pre čitateľa, ktorý
> o dopravnom modelovaní nič nevie. Technická dokumentácia (inštalácia, príkazy,
> Docker) je v [README.md](README.md).

---

## Obsah

- [Čo tento projekt vlastne robí](#čo-tento-projekt-vlastne-robí)
- [Základná myšlienka: štyri kroky klasického dopravného modelu](#základná-myšlienka-štyri-kroky-klasického-dopravného-modelu)
- [Slovníček pojmov](#slovníček-pojmov)
- [Ako sa to spúšťa](#ako-sa-to-spúšťa)
- [Krok za krokom, čo sa deje](#krok-za-krokom-čo-sa-deje)
- [Odkiaľ sa berie počet áut na vstupe — a čo je na ňom „nedeterministické"](#odkiaľ-sa-berie-počet-áut-na-vstupe--a-čo-je-na-ňom-nedeterministické)
- [Ako čítať validačné metriky](#ako-čítať-validačné-metriky)
- [Scenáre — čo model reálne umožňuje](#scenáre--čo-model-reálne-umožňuje)
- [Ako je repozitár organizovaný](#ako-je-repozitár-organizovaný)
- [Čo je na tom v skutočnosti ťažké](#čo-je-na-tom-v-skutočnosti-ťažké)

---

## Čo tento projekt vlastne robí

Je to **simulátor automobilovej dopravy v meste**. Odpovedá na otázku:

> „Koľko áut denne prejde po každej jednej ulici v Brne — a čo sa stane, keď
> niektorú z nich zavriem?"

Nie je to hra ani vizualizácia jednotlivých áut. Je to **makroskopický model**:
nesleduje jednotlivé autá, ale *toky* — ako voda v potrubí. Každý úsek cesty
dostane číslo „koľko vozidiel tadiaľto prejde za deň".

Celé je to diplomová práca a je to postavené na knižnici
**[AequilibraE](https://www.aequilibrae.com/)** (open-source dopravný modelovací
nástroj), ktorá rieši matematické jadro. Kód v tomto repozitári hlavne **zháňa a
pripravuje dáta**, kŕmi nimi AequilibraE, a potom výsledok **kalibruje podľa
reálnych meraní**.

---

## Základná myšlienka: štyri kroky klasického dopravného modelu

Doprava sa v odbore modeluje už 60 rokov rovnakým receptom (tzv. *four-step
model*). Celý tento repozitár je v podstate implementácia tohto receptu:

| Krok | Otázka | Kde v kóde |
|---|---|---|
| 1. **Generovanie** | Koľko ciest denne vznikne v každej štvrti? | [src/sim/demand/](src/sim/demand/) |
| 2. **Distribúcia** | Odkiaľ kam tie cesty idú? | [src/sim/distribution/](src/sim/distribution/) |
| 3. **Voľba módu** | Autom, MHD, pešo? | *preskočené* — model rieši len autá, podiel áut je konštanta (`car_share`) |
| 4. **Priradenie** | Ktorou trasou pôjdu? | [src/sim/assignment/](src/sim/assignment/) |

Plus dve veci navyše, ktoré klasický recept nemá a ktoré tvoria polovicu tohto kódu:

- **automatické zháňanie dát** (OSM, ČSÚ, ŘSD) — [src/sim/datasets/](src/sim/datasets/)
- **kalibrácia** — dolaďovanie modelu, aby sedel s reálnymi sčítaniami áut — [src/sim/calibration/](src/sim/calibration/)

---

## Slovníček pojmov

Bez nich sa ďalej nepohneme.

**Link (hrana)** — jeden úsek cesty medzi dvoma križovatkami. Brnenská sieť ich
má desiatky tisíc. Každý má dĺžku, počet pruhov, rýchlosť, kapacitu.

**Node (uzol)** — križovatka alebo koniec úseku.

**TAZ / zóna** — *Traffic Analysis Zone*. Mesto sa rozdelí na kúsky (tu: mestské
časti z OpenStreetMap, `admin_level: 9`). Model nevie, že bývate na Kounicovej 12
— vie len, že bývate v zóne „Brno-Královo Pole".

**Centroid** — jeden umelý bod v strede zóny. Všetka doprava zo zóny „vzniká"
v tomto bode.

**Konektor** — umelá cesta, ktorou sa centroid pripája na reálnu cestnú sieť. Bez
nej by autá zo zóny nemali kadiaľ vyjsť. Tvoria sa v
[src/sim/zoning/connectors.py](src/sim/zoning/connectors.py).

**OD matica** — *Origin–Destination*. Tabuľka N×N (zóna × zóna), kde bunka
`[i][j]` = koľko áut denne ide zo zóny `i` do zóny `j`. Toto je **srdce celého
modelu**. Pre Brno má stovky riadkov a stĺpcov.

**Skim** — matica *časov jazdy* medzi zónami (na rozdiel od OD matice, kde sú
počty ciest). „Ako dlho trvá z Bohuníc do Líšne."

**Gateway (brána)** — miesto, kde diaľnica/cesta preteká hranicou modelu
(D1, D2, I/43, I/52 pri Brne). Doprava zvonku sa musí niekde „naliať" dnu. Kód pre
ne vyrába **umelé zóny** s ID od 8 000 000 000.

**Screenline** — pomyselná čiara naprieč mestom. Sčíta sa, koľko áut ju v modeli
prekročí, a porovná sa s realitou. Kontrolný bod kalibrácie.

**BPR funkcia** — vzorec, ktorý hovorí *o koľko sa cesta spomalí, keď je
preplnená*:

```
čas = čas_naprázdno × (1 + α × (objem / kapacita)^β)
```

Pri α=0.55, β=4: keď je cesta na 100 % kapacity, jazda trvá 1.55× dlhšie. Pri
150 % už 3.8× dlhšie. Parametre α, β sú per typ cesty v
[src/sim/defaults.py:148](src/sim/defaults.py#L148) — diaľnica má α=0.15 (znesie
preťaženie lepšie), obytná ulica α=0.85.

**V/C (VOC)** — objem/kapacita. Nad 1.0 = kolóna.

**LOS** — *Level of Service*, známka A–F podľa V/C. A = voľno, F = stojíte.
[src/sim/assignment/executor.py:313](src/sim/assignment/executor.py#L313)

**GEH** — dopravná štatistika na porovnanie modelu s meraním. Nie je to obyčajný
rozdiel — toleruje väčšie odchýlky pri veľkých číslach. GEH < 5 = „dobrá zhoda".

**CSD** — Celostátní sčítání dopravy, ŘSD ho robí každých 5 rokov. Reálne namerané
počty áut na cestách. **Toto je pravda, s ktorou sa model porovnáva.**

**SLDB 2021** — Sčítání lidu. ČSÚ z neho zverejňuje tabuľku „koľko ľudí dochádza
z obce X do obce Y za prácou/školou". **Toto je hlavný zdroj toho, kam ľudia
cestujú.**

---

## Ako sa to spúšťa

Všetko ide cez jeden súbor — [run.py](run.py). Nie je to jeden dlhý beh, ale
**20 samostatných krokov**, ktoré sa púšťajú postupne:

```bash
python run.py --config config/brno/sim.yaml build-network
python run.py --config config/brno/sim.yaml fetch-data
...
```

`run.py` je v podstate rozcestník: skontroluje, či existujú vstupné súbory pre
daný krok ([`_STEP_PREREQUISITES`](run.py#L158)), varuje, ak sú výstupy zastarané
([`_STALENESS_CHECKS`](run.py#L204)), a zavolá príslušnú funkciu.

**Prečo po krokoch a nie naraz?** Lebo niektoré kroky bežia desiatky minút a keď
doladíte jeden parameter, chcete prepočítať len tú časť, ktorej sa to týka.

---

## Krok za krokom, čo sa deje

> Technický rozbor každého kroku — vstupy, algoritmy, rizikové miesta, nálezy
> v kóde a návrhy na zlepšenie — je v samostatnom dokumente
> [pipeline_detail.md](pipeline_detail.md).

### 1. `build-network` — stiahnutie cestnej siete

[src/sim/network/pipeline.py](src/sim/network/pipeline.py)

1. Z konfigu sa vezme `place_name: "Brno, Czechia"`, geokóduje sa na polygón.
2. AequilibraE stiahne z OpenStreetMap všetky cesty v tejto oblasti (+ 5 km
   rezerva na okrajoch, aby sa nepotrhali napojenia).
3. **Vyhodia sa nezjazdné cesty** — chodníky, cyklotrasy, schody
   ([filtering.py](src/sim/network/filtering.py)).
4. **Vyhodia sa odrezané ostrovčeky** — kúsky siete, ktoré nie sú napojené na
   hlavný celok (typicky chyby v OSM). Ostane len najväčší súvislý komponent.
5. **Oreže sa na mestské jadro** a znova sa skontroluje súvislosť.
6. **Obohatenie z OSM** ([osm_enrichment.py](src/sim/network/osm_enrichment.py))
   — dotiahnu sa značky, ktoré AequilibraE pri importe zahodí: `maxspeed`,
   `ref` (číslo cesty ako „D1"), názov ulice.
7. Vypadnú PNG mapy a GeoJSON do `outputs/brno/maps/`.

**Výsledok:** SQLite databáza `project/brno_aeq/project_database.sqlite`
s tabuľkami `links` a `nodes`.

### 2. `fetch-data` — stiahnutie externých dát

[src/sim/datasets/pipeline.py](src/sim/datasets/pipeline.py)

Rozposiela sa podľa typu zdroja („provider"):

- **ČSÚ SLDB 2021** — CSV s dochádzkou do práce/školy medzi obcami → parquet
- **ŘSD CSD 2025** — XLSX so sčítaním dopravy → parquet
- **populácia po zónach** — CSV → napárované na zóny
- **ATOM feed ČÚZK** — centroidy všetkých obcí ČR
- **PostgreSQL** — uzávierky (NDIC/Polícia), Waze zápchy, dopravné udalosti

Národné dáta idú do zdieľaného `data/sources/` — sťahujú sa raz pre všetky mestá.

Navyše sa tu odvodí **zamestnanosť po zónach**: keď z dochádzkovej tabuľky
spočítate, koľko ľudí *prichádza* do zóny X, dostanete odhad počtu pracovných
miest tam. ([employment.py](src/sim/datasets/employment.py))

### 3. `normalize-network` — dolaďovanie parametrov ciest

[src/sim/network/normalization.py](src/sim/network/normalization.py) — jeden
z najdôležitejších a najviac dolaďovaných súborov.

OSM dáta sú deravé: chýbajú rýchlosti, počty pruhov, kapacity. Tu sa doplnia.

**Rýchlosť — dve vrstvy:**

- *Posted speed* (nominálna): buď z OSM, alebo z tabuľky podľa typu cesty
- *Practical speed* (reálna):
  [`_apply_practical_speed_reduction`](src/sim/network/normalization.py#L282)

  ```
  praktická = posted × base_factor − penalizácia_za_km × hustota_križovatiek
  ```

  Lebo po ulici s limitom 50 s križovatkou každých 100 m nikdy nejdete 50.
  Diaľnica má vysoký `base_factor` a nízku penalizáciu, obytná ulica naopak.

  Dôležitý detail: pôvodná rýchlosť sa uloží do stĺpca `posted_speed_ab`, aby
  opakované spustenie kroku rýchlosť neznižovalo stále dokola (idempotencia).

**Kapacita:** `kapacita = kapacita_na_pruh × počet_pruhov`, plus
[„CSD hints"](src/sim/network/normalization.py#L83) — ak reálne meranie hovorí,
že tadiaľ prejde 40 000 áut denne, kapacita nesmie byť menšia než ~10 % z toho.

**Oprava „pinch pointov"**
([`_fix_lane_pinch_points`](src/sim/network/normalization.py#L152)): OSM často
zmapuje prechod z obojsmernej cesty na diaľnicu jedným krátkym jednopruhovým
úsekom. Model by tam videl umelé úzke hrdlo. Kód to nájde a rozšíri.

**Čas jazdy naprázdno:** `čas = vzdialenosť × 3.6 / rýchlosť`

Všetko sa zapíše späť do SQLite, oddelene pre smer A→B a B→A (jednosmerky,
asymetrické počty pruhov).

### 4. `build-zones` — rozdelenie mesta na zóny

[src/sim/zoning/pipeline.py](src/sim/zoning/pipeline.py)

1. Načítajú sa hranice mestských častí z OSM (okres Brno-město + Brno-venkov).
2. Odfiltrujú sa tie, ktorých ťažisko je mimo modelovanej oblasti.
3. Odstránia sa prekryvy (dva zdroje môžu popisovať to isté územie).
4. **Vytvoria sa umelé „gateway" zóny**
   ([gateways.py](src/sim/zoning/gateways.py)) na miestach, kde D1/D2/I/43/I/52
   pretínajú hranicu. V configu Brna sú niektoré kotvené na presné súradnice,
   lebo automatika trafila vedľajšiu vetvu.
5. Vypočítajú sa **centroidy** a **konektory** — pre každú zónu až 6 spojok na
   najbližšie vhodné cesty. Interné zóny sa zámerne nenapájajú priamo na diaľnicu.
6. Priradí sa **populácia** ku každej zóne.
7. Uloží sa mapovanie `zóna → ID centroid uzla` (`zone_centroid_mapping.json`) —
   to potrebuje takmer každý ďalší krok.

### 5. `build-supernetwork` — tranzitná doprava

[src/sim/supernetwork/pipeline.py](src/sim/supernetwork/pipeline.py) —
najmenej intuitívny krok.

**Problém:** cez Brno prechádza doprava, ktorá v Brne nič nemá — napr.
Praha → Ostrava po D1. Model o nej nevie, lebo dochádzková tabuľka
Praha→Ostrava sa Brna „netýka". Ale tie autá cez Brno reálne prejdú.

**Riešenie:** postaví sa **hrubá sieť celej ČR** (len diaľnice, cesty I. a II.
triedy). Potom:

1. Napojí sa (snap) každá brnenská brána a centroid každej obce ČR na túto sieť.
2. Vypočítajú sa najkratšie cesty medzi bránami a obcami.
3. Pre každý dochádzkový vzťah v ČR sa určí, či trasa vedie cez Brno, a
   klasifikuje sa ([classification.py](src/sim/supernetwork/classification.py)):
   - `internal` — v rámci Brna
   - `external_internal` / `internal_external` — dnu/von
   - `external_external` — **tranzit**, prejde naprieč
4. Filtruje sa: tranzit sa uzná, len ak zachádzka cez Brno nie je väčšia než
   25 % ([`detour_ratio_max`](src/sim/defaults.py#L113)) — inak by tade nikto
   nešiel.

**Výstupom** sú dve tabuľky: `external_gateway_lookup.parquet` (ktorou bránou
vstupuje doprava z ktorej obce) a `through_gateway_pairs.parquet` (koľko áut
denne ide z brány A do brány B).

### 6. `build-demand` — zostavenie OD matice

[src/sim/demand/pipeline.py](src/sim/demand/pipeline.py)

Skladá sa niekoľko vrstiev („segmentov") dopravy:

| Segment | Odkiaľ | Ako |
|---|---|---|
| `commuting` | SLDB dochádzka | osoby → autá: `osoby × podiel_áut / obsadenosť × ciest_na_osobu` |
| `other` | populácia | gravitačný model — [`_build_gravity_seed`](src/sim/demand/seeds.py#L21) |
| `external_local` | brány ↔ mesto | pevný denný objem z configu (Brno: 35 334), rozdelený podľa populácie zón |
| `external_through` | supernetwork | tranzit z kroku 5, škálovaný ×0.30 |

**Gravitačný model** je fyzikálna analógia: cesty medzi dvoma zónami sú úmerné
súčinu ich „hmotností" (populácia) a klesajú so vzdialenosťou:

```
cesty[i][j] = produkcia[i] × atrakcia[j] × e^(−β × vzdialenosť)
```

Všetko sa uloží ako `.aem` súbor (AequilibraE matica), s viacerými „cores"
(vrstvami) — zvlášť commuting, zvlášť other, zvlášť tranzit, a súčtový `wd_daily`.

### 7. `assign-warm-skims` — rýchly predbeh

Aby ďalší krok vedel, ako *dlho* trvá cesta medzi zónami, musí sa už raz niečo
priradiť na sieť. Toto je krátke, nepresné priradenie (30 iterácií), z ktorého sa
vezme len matica časov (`skims.aem`).
[src/sim/assignment/pipeline.py:285](src/sim/assignment/pipeline.py#L285)

### 8. `distribute` — prerozdelenie ciest

[src/sim/distribution/pipeline.py](src/sim/distribution/pipeline.py)

Hrubý odhad z kroku 6 sa opraví, aby sedel na reálne cestovné časy a na kapacity
zón:

1. **Kalibrácia gravitačného modelu** — z hrubej matice a matice časov sa fitne
   parameter β (odpor voči vzdialenosti).
   [gravity.py:9](src/sim/distribution/gravity.py#L9)
2. **P/A vektory** — pre každú zónu:
   - *production* (koľko ciest vzniká) = f(populácia)
   - *attraction* (koľko ciest priťahuje) = f(zamestnanosť)

   Ak chýbajú dáta o zamestnanosti, kód **zámerne padne s chybou**
   ([pipeline.py:163](src/sim/distribution/pipeline.py#L163)) — bez nich by boli
   atrakcie symetrickou kópiou produkcií, čo je štrukturálne nesprávne.
3. **IPF / Furness** ([gravity.py:62](src/sim/distribution/gravity.py#L62)) —
   iteratívne škálovanie riadkov a stĺpcov matice, kým súčty nesedia na cieľové
   P/A. Striedavo: „vydeľ riadky, aby sedeli súčty riadkov" → „vydeľ stĺpce" →
   dokola, kým to nekonverguje.
4. **Blend** — výsledok sa zmieša s pôvodným odhadom v pomere daným
   `blend_alpha` (default 0.55 → 55 % IPF, 45 % pôvodný seed), aby sa nestratila
   informácia zo SLDB. Výsledok sa ešte zastropuje na `max_total_multiplier`
   (3.5× seed), aby chýbajúce dáta o zamestnanosti maticu nenafúkli.

### 9. `assign` — priradenie na sieť ⭐

[src/sim/assignment/executor.py](src/sim/assignment/executor.py) — jadro celej
simulácie.

Otázka: *máme maticu „kto kam ide" a sieť ciest — ktorou trasou pôjdu?*

Naivná odpoveď: „každý najrýchlejšou trasou" (all-or-nothing). To je **zle** a kód
to explicitne zakazuje
([executor.py:34](src/sim/assignment/executor.py#L34)) — keby všetci šli po tej
istej najrýchlejšej ceste, bola by upchatá a už by nebola najrýchlejšia.

Správna odpoveď: **užívateľské ekvilibrium** (Wardropov princíp) — hľadá sa taký
stav, kde *nikto si nemôže polepšiť zmenou trasy*. Rieši sa iteratívne algoritmom
**BFW** (biconjugate Frank-Wolfe):

1. Priraď všetkých na najrýchlejšie trasy
2. Prepočítaj časy podľa BPR (preplnené cesty sa spomalili)
3. Presuň časť dopravy na nové najrýchlejšie trasy
4. Opakuj, kým sa „relative gap" (miera nerovnováhy) nedostane pod 0.001

**Dôležitá finta — denná kapacita**
([graph.py:73](src/sim/assignment/graph.py#L73)): matica je *denná*, ale kapacity
ciest sú *hodinové*. Tak sa kapacita vynásobí faktorom ~10–13 (diaľnica 13,
obytná ulica 8). Inverzia tohto faktora je zároveň „K-faktor" na spätný prepočet
špičkovej hodiny.

**Multi-class** ([defaults.py:180](src/sim/defaults.py#L180)): lokálna a
tranzitná doprava sa priradzujú ako dve samostatné triedy. Tranzit má `vot: 2.0`
(viac si cení čas → menej zachádza) a `pce: 1.5` (viac nákladiakov → viac zaťaží
cestu).

**Výstup:** `assignment_results.parquet` — pre každý link objem, V/C, LOS,
špičková hodina, zdržanie.

### 10. `audit-supply` — kontrola „ponuky"

[src/sim/calibration/supply_audit.py](src/sim/calibration/supply_audit.py)

Skontroluje, či rýchlosti/kapacity/BPR parametre dávajú zmysel. Je to **povinná
brána pred kalibráciou** ([context.py:72](src/sim/calibration/context.py#L72)) —
s odôvodnením priamo v kóde: *ODME nesmie kompenzovať chyby na strane siete*.
Keby ste mali zle nastavené kapacity, kalibrácia by to „opravila" pokrivením
dopytu, čo je zamaskovanie chyby, nie jej odstránenie.

### 11. `calibrate` — ODME ⭐

[src/sim/calibration/odme.py](src/sim/calibration/odme.py) +
[context.py](src/sim/calibration/context.py)

**ODME** = *Origin-Destination Matrix Estimation*. Toto je najzložitejšia časť.

**Problém:** model predpovie 25 000 áut na Kounicovej, reálne sčítanie hovorí
32 000. Kde je chyba? Nevieme — chyba je niekde v OD matici. Ale ktorú z tisícok
jej buniek treba zväčšiť?

**Riešenie (Spiessova gradientná metóda):**

1. Spusti priradenie s aktuálnou maticou.
2. Napáruj namerané sčítacie stanoviská na linky v sieti
   ([matching.py](src/sim/calibration/matching.py) — netriviálne: geometrické
   buffery, kontrola smeru, agregácia koridoru, kontrola čísla cesty).
3. Spočítaj **účelovú funkciu**:

   ```
   Z = Σ w × (model − realita)²
   ```

   kde váha `w = 1/√(objem)` — inak by cieľu dominovali diaľnice a malé ulice by
   sa ignorovali. [context.py:180](src/sim/calibration/context.py#L180)
4. Pre každú screenline zisti pomer `realita / model`.
5. **Select-link analýza** — AequilibraE povie, *ktoré OD páry* cez danú
   screenline reálne prechádzajú. To je kľúč: opravujú sa len tie bunky matice,
   ktoré danú screenline používajú, úmerne ich podielu.
   [odme.py:27](src/sim/calibration/odme.py#L27)

   ```python
   adjustment = 1 + damping × (ratio − 1) × proportion
   demand *= adjustment
   ```
6. Aplikuj ešte globálnu korekciu, korekciu po triedach ciest a kalibráciu brán.
7. **Zastrihni** — žiadna bunka sa nesmie odchýliť od pôvodného odhadu viac než
   `max_deviation` (4×). Toto je poistka proti tomu, aby kalibrácia nezničila
   informáciu zo SLDB.
8. Späť na bod 1, max 48× (Brno).

Loop má viacero ochranných mechanizmov:

- **damping** sa automaticky zmenší pri zhoršení a obnoví po dvoch zlepšeniach
- pri **3 zhoršeniach za sebou** sa vráti najlepšia matica (`REVERT`)
- **stall detection** — ak sa 20 iterácií nezlepšilo, končí
- **stages** — Brno má dve fázy: „stabilize" (20 iterácií) a „refine" (35)

Konvergenčné kritériá pre denný model
([context.py:765](src/sim/calibration/context.py#L765)): R² ≥ 0.80, sklon
regresie v pásme 0.85–1.15, %RMSE ≤ 35, odchýlka ≤ 15 %. GEH sa **zámerne
nepoužíva** — je navrhnutý pre hodinové toky.

### 12. `validate` — nezávislé overenie

[src/sim/calibration/validation.py](src/sim/calibration/validation.py)

Zásadné pravidlo vedeckej poctivosti: **nesmiete sa chváliť, že model sedí na
dátach, na ktorých ste ho ladili.**

Preto sa CSD dáta hneď na začiatku rozdelia 65/35
([observed.py:441](src/sim/calibration/observed.py#L441)):

- 65 % → kalibrácia
- 35 % → **holdout**, model ich nikdy nevidel

Validácia beží len na holdoute a počíta R², %RMSE, GEH, bias, MAE po triedach
ciest, Spearmanovu koreláciu + scatter plot. Kód dokonca ukladá SHA-hash
kalibračných úsekov, aby sa dalo overiť, že sa množiny neprekryli
([odme.py:548](src/sim/calibration/odme.py#L548)).

Rieši sa tu aj nepríjemný detail: **rozdelené štvorpruhovky**. ŘSD udáva pre
diaľnicu jedno číslo (súčet oboch smerov), ale OSM ju má ako dva jednosmerné
linky. Kód to deteguje a spočíta
([validation.py:236](src/sim/calibration/validation.py#L236)).

### 13.–15. `learn-profile`, `strip-closures`, `serve`

- **`learn-profile`** ([temporal.py](src/sim/demand/temporal.py)) — z CSD sa
  naučia koeficienty pre typ dňa (pracovný / sobota / nedeľa / sviatok)
  a rozdelenie deň/večer/noc. Umožňuje z dennej matice odvodiť ľubovoľný deň.
- **`strip-closures`** — odstráni z modelu uzávierky, ktoré platili v čase
  kalibrácie, aby scenáre vychádzali z čistej siete.
- **`serve`** ([api.py](src/sim/api.py)) — spustí FastAPI server (~28 endpointov)
  a React frontend (git submodul) z neho kreslí mapy a reporty.

---

## Odkiaľ sa berie počet áut na vstupe — a čo je na ňom „nedeterministické"

### Krátka odpoveď

**V ceste k počtu áut nie je žiadny generátor náhodných čísel.** Jediné volanie
RNG v celom `src/sim/` je rozdelenie sčítacích úsekov na kalibračnú a holdout
časť ([observed.py:540](src/sim/calibration/observed.py#L540)) a to má
napevno zafixovaný seed (`random_seed: 42`). Model je makroskopický — nemodeluje
jednotlivé autá, takže nikde nič nevzorkuje. Dva behy `build-demand` nad tými
istými dátami a tým istým configom dajú **bit po bite tú istú maticu**.

Čo je na tom „nedeterministické" v bežnom (nie matematickom) zmysle:

> Počet áut na vstupe **nie je nameraný — je odhadnutý.** Skladá sa zo štyroch
> vrstiev, každá má voľné parametre zvolené modelárom, a v každom ďalšom kroku
> pipeline sa to číslo ešte posunie. Nakoniec ho kalibrácia prepíše podľa
> reálnych sčítaní.

Sú v tom tri rôzne veci, ktoré sa oplatí rozlišovať:

| | Typ | Čo to je | Reprodukovateľné? |
|---|---|---|---|
| **A** | **Voľné parametre** | `car_share`, `occupancy`, `trip_rate`, `beta`, škálovacie faktory | Áno — ale iná voľba = iné číslo |
| **B** | **Skutočná náhoda (RNG)** | rozdelenie sčítaní kalibrácia/holdout | Áno, seed = 42 |
| **C** | **Drift dát v čase** | OSM, ČSÚ, ŘSD, uzávierky sa sťahujú zo živých zdrojov | Nie — ten istý príkaz o mesiac dá iné čísla |

Drvivá väčšina toho, čo vyzerá ako nedeterminizmus, je **A** — deterministický
výpočet z arbitrárne zvolených konštánt.

### Ako sa to číslo mení krok za krokom (reálny beh, Brno)

Čísla nižšie sú z reálneho behu uloženého v
[simulation_for_article/updated_version_1/](simulation_for_article/updated_version_1/)
(`od_summary.json`, `calibration_report.json`):

| Po kroku | Celkový počet vozidlojázd / deň | Zmena |
|---|---:|---|
| `build-demand` (surová OD matica) | **461 374** | — |
| `distribute` (gravitácia + IPF + blend) | **532 412** | +15,4 % |
| `calibrate` (ODME, iterácia 36 z 48) | **384 869** | −27,7 % oproti distribúcii |

Čiže: číslo, ktoré vojde do modelu, sa medzi prvým a posledným krokom zmení
o desiatky percent — a to bez jediného náhodného čísla. To je jadro toho, čo
myslel Adam.

### Krok po kroku: čo určuje počet áut a koľko voľnosti tam je

#### 1. `fetch-data` — SLDB dochádzka (vstup sú **osoby**, nie autá)

Zo sčítania ČSÚ 2021 sa berie tabuľka „dojížďka do zaměstnání a škol" — počet
*osôb*, ktoré dochádzajú z obce A do obce B. To je tvrdé dáto, nie odhad.

Voľnosti (typ A):
- `demand.sldb.include_lokalizace` — ktoré typy dochádzky sa započítajú
  (default `0_na_adrese_OP`, `1_meziobecni`). Zapnutie/vypnutie kategórie mení
  vstup skokovo.
- Dáta sú z roku **2021** (covidový rok, sčítanie k 26. 3. 2021), model beží na
  siete a sčítaniach z roku **2025**. Rozdiel sa nikde explicitne nekompenzuje —
  „dorovná" ho až kalibrácia.

#### 2. `build-demand` — konverzia osôb na autá (tu vzniká najviac voľnosti)

Štyri vrstvy, každá s vlastným vzorcom
([pipeline.py](src/sim/demand/pipeline.py), [seeds.py](src/sim/demand/seeds.py)):

**a) `commuting` — 297 113 vozidiel/deň**

```
vozidlá = osoby × car_share / occupancy × trips_per_person
```

s hodnotami z [defaults.py:224](src/sim/defaults.py#L224):

| Účel | `car_share` | `occupancy` | `trips_per_person` | výsledný prepočet |
|---|---:|---:|---:|---|
| práca | 0,48 | 1,20 | 2,0 | 1 osoba = **0,80** vozidlojazdy/deň |
| škola | 0,25 | 1,30 | 2,0 | 1 osoba = **0,38** vozidlojazdy/deň |

Vzťah je **lineárny**, takže citlivosť je triviálna: `car_share` 0,48 → 0,55
znamená +14,6 % áut v tomto segmente (≈ +43 000 vozidiel/deň). Podobne
`occupancy` 1,20 → 1,35 znamená −11 % (≈ −33 000). Žiadna z týchto hodnôt nie je
nameraná pre Brno — sú to hodnoty z literatúry/odhad.

Navyše sa dochádzka **cez hranicu modelu** (externá obec ↔ Brno) násobí
`external_commuting_scale = 0,65`
([defaults.py:219](src/sim/defaults.py#L219)) — predpoklad, že nie každý dochádzajúci
z celej ČR jazdí denne. Ďalší voľný parameter.

**b) `other` (nákup, voľný čas, služobné cesty) — 93 035 vozidiel/deň**

Nemá dátový zdroj vôbec, generuje sa gravitačným modelom z populácie
([`_build_gravity_seed`](src/sim/demand/seeds.py#L21)):

```
produkcia[i] = populácia[i] × trip_rate / occupancy × car_share
cesty[i][j]  = produkcia[i] × populácia[j] × e^(−β × vzdialenosť)
```

s `trip_rate = 1.0`, `car_share = 0.35`, `occupancy = 1.50`, `beta = 0.00030`
([defaults.py:231](src/sim/defaults.py#L231)). Celý tento segment (20 % dopravy) je
teda **čistý predpoklad** — zmena `trip_rate` z 1,0 na 1,5 pridá do modelu
~46 500 áut denne.

**c) `external_local` (brána ↔ mesto) — 35 334 vozidiel/deň**

Buď pevné číslo z configu, alebo `total_daily_trips: auto`, kedy sa odhadne ako
súčet CSD AADT na cestách, na ktorých ležia brány, delený pokrytím
([`_estimate_total_daily_trips_from_csd`](src/sim/demand/seeds.py#L132)).

Toto je najlepšia ilustrácia typu A: v [config/brno/sim.yaml](config/brno/sim.yaml)
je komentár, že predchádzajúca hodnota **130 000** bola odhad bez opory v dátach
a nahradila ju hodnota **35 334** odvodená z kamdojizdime.cz. To je zmena vstupu
o −95 000 áut denne — bez zmeny jediného riadku kódu.

**d) `external_through` (tranzit) — 35 891 vozidiel/deň**

Zo supernetworku (krok 5), vynásobené `through_traffic_scale`
— v Brne **0,30**, default v kóde **0,50**
([defaults.py:218](src/sim/defaults.py#L218)). Tento jediný parameter mení tranzit
o desiatky percent a je to zároveň jediný parameter, ktorý má v repozitári
predpripravený citlivostný sweep (`[0.25, 0.50, 0.75, 1.00]`,
[defaults.py:490](src/sim/defaults.py#L490)).

**e) Rozdelenie do období dňa**

`am/ip/pm/ev` podiely (napr. `other`: 0.25/0.30/0.30/0.15) sú tiež pevne zvolené
konštanty. Denný súčet nemenia, menia však, kedy sieť „zahustne", a teda cez BPR
aj výsledné časy.

#### 3. `distribute` — prvý veľký posun celkového čísla (+15 %)

Tu sa počet áut **nezachováva**. IPF ťahá maticu na cieľové P/A vektory
spočítané z populácie a zamestnanosti *nezávisle* od kroku 2, s ďalšou trojicou
parametrov: `pa_trip_rate = 1.8`, `pa_car_share = 0.40`, `pa_occupancy = 1.3`
([defaults.py:288](src/sim/defaults.py#L288)). Výsledok sa mieša so seedom
(`blend_alpha = 0.55`) a stropuje (`max_total_multiplier = 3.5`).

Kombinácia „vlastné P/A vektory + blend + strop" je dôvod, prečo z 461 374
vyjde 532 412. Aj tu je všetko deterministické — a všetko sú zvolené konštanty.

#### 4. `assign` — počet áut sa nemení, mení sa ich rozloženie

BFW iteruje, kým relative gap neklesne pod `rgap_target` (0.002 pri hlavnom
priradení, 0.008 vnútri ODME). To je jediné miesto, kde vzniká **výpočtová**
neurčitosť dvoch druhov:

- **tolerancia konvergencie** — riešenie nie je presné ekvilibrium, len bod
  „dosť blízko"; pri rgap 0,002 sa objemy na jednotlivých linkoch môžu líšiť
  o jednotky percent od exaktného riešenia, hoci celkový počet áut je rovnaký;
- **viacvláknové sčítanie** — priradenie beží na všetkých jadrách
  ([executor.py:217](src/sim/assignment/executor.py#L217)), takže poradie sčítania
  float čísel nie je zaručené a dva behy sa môžu líšiť na posledných desatinných
  miestach. Prakticky zanedbateľné, ale technicky to je jediná vec, ktorá robí
  výstup bit-po-bite nereprodukovateľným.

#### 5. `calibrate` (ODME) — druhý veľký posun (−28 %)

Tu sa vstupný počet áut **prepisuje podľa reálnych meraní**, a to tromi
mechanizmami:

1. **Kalibrácia brán** ([context.py:853](src/sim/calibration/context.py#L853)) — objem
   na každom vstupnom koridore sa vynásobí faktorom v rozsahu 0,60–1,45,
   tlmeným damping faktorom 0,20, aby sedel na pozorované AADT.
2. **Spiessov gradient** — každá bunka OD matice sa môže pohnúť len v pásme
   `seed / 4` až `seed × 4` (`max_deviation = 4.0`,
   [context.py:449](src/sim/calibration/context.py#L449)). Toto je „elasticita" —
   povolená miera, o koľko smie kalibrácia prepísať vstupný odhad.
3. **Strop na iteráciu** — jedna iterácia nesmie zmeniť celkový dopyt o viac ako
   ±12 % (`max_iter_change_pct`,
   [context.py:885](src/sim/calibration/context.py#L885)); z 48 iterácií sa nakoniec
   vyberie najlepšia (v ukážkovom behu č. 36).

Dôsledok, ktorý je dôležitý pre obhajobu: **konečný počet áut v modeli je
z veľkej časti určený sčítaniami ŘSD, nie vstupnými predpokladmi.** Predpoklady
z kroku 2 určujú *tvar* matice (kto kam ide) a východiskový bod; kalibrácia
určuje *mieru*. Preto zmena `car_share` o 15 % neposunie výsledné objemy na
linkoch o 15 % — kalibrácia väčšinu toho pohltí. Ale len tam, kde sú merania:
na 14 použiteľných CSD úsekoch a bránach. Vo štvrtiach bez sčítania zostáva
vstupný predpoklad nedotknutý.

#### 6. `validate` — jediné miesto so skutočnou náhodou

Sčítacie úseky sa rozdelia na kalibračných 65 % a holdout 35 %
stratifikovaným náhodným výberom
([observed.py:540](src/sim/calibration/observed.py#L540)):

```python
rng = np.random.RandomState(random_seed)   # random_seed = 42
road_info = road_info.sample(frac=1.0, random_state=rng)
```

Seed je fixný, takže rozdelenie je reprodukovateľné. **Ale je arbitrárne** —
so `random_seed: 7` by do kalibrácie išli iné úseky, ODME by dostalo iné ciele
a celkový počet áut v modeli by vyšiel iný. Pri 14 použiteľných úsekoch (viď
[Prečo je použiteľných len 14 z desiatok CSD úsekov](#prečo-je-použiteľných-len-14-z-desiatok-csd-úsekov))
je táto citlivosť nezanedbateľná — je to najmenšie miesto v pipeline s najväčším
pákovým efektom. Ak treba obhájiť robustnosť, správna odpoveď je prebehnúť
kalibráciu s viacerými seedmi a ukázať rozptyl výsledných metrík.

### Zhrnutie: čo odpovedať na otázku „je to nedeterministické?"

1. **Nie v zmysle náhody.** Pipeline nemá stochastické priradenie, žiadny Monte
   Carlo, žiadne vzorkovanie jednotlivých vozidiel. Rovnaký config + rovnaké
   dáta = rovnaký výsledok (až na float šum z viacvláknového priradenia).
2. **Áno v zmysle neurčitosti vstupu.** Počet áut nie je meraný; je poskladaný
   z ~12 voľných parametrov (`car_share`, `occupancy`, `trip_rate`,
   `external_commuting_scale`, `through_traffic_scale`, `total_daily_trips`,
   `beta`, `pa_*`, `blend_alpha`, podiely období). Ich zmena mení vstup lineárne
   a v niektorých prípadoch o desiatky percent.
3. **Áno v zmysle jedného náhodného rozhodnutia**, ktoré je zafixované seedom 42
   — rozdelenie sčítaní na kalibráciu a holdout.
4. **Áno v zmysle času.** OSM, CSD, uzávierky a Waze sa sťahujú zo živých
   zdrojov; ten istý príkaz spustený o pol roka postaví inú sieť aj iné ciele
   kalibrácie. Preto sú výsledky pre článok zamrazené v
   [simulation_for_article/](simulation_for_article/).
5. **Kalibrácia väčšinu neurčitosti z bodu 2 pohltí** — ale len na linkoch, kde
   existuje meranie. Rozdiel medzi kalibračnou a holdout časťou reportu je presne
   tá metrika, ktorá hovorí, koľko z toho zostalo.

---

## Ako čítať validačné metriky

Krok `validate` vypíše tabuľku čísel a verdikt PASS/FAIL. Tu je, čo každé číslo
znamená, prečo tam je a aký má prah.

Všetky metriky porovnávajú **dva stĺpce čísel**:

- **observed** (`O`) — reálne namerané vozidlá z CSD na danom úseku
- **modeled** (`M`) — čo na tom istom úseku predpovedal model

Počítajú sa v [src/sim/calibration/metrics.py:43](src/sim/calibration/metrics.py#L43)
(`compute_stats`).

### Prehľad — čo ktorá metrika zachytí

Žiadna metrika sama o sebe nestačí. Každá je slepá voči inej chybe:

| Metrika | Odpovedá na otázku | Slepá voči |
|---|---|---|
| **R²** | Sedí *poradie* — sú veľké ulice veľké? | systematickému posunu |
| **slope** | Nie sú všetky čísla systematicky pod/nad? | rozptylu |
| **bias** | Sedí *celkový objem* siete? | tomu, či sedí rozdelenie |
| **%RMSE** | Aký veľký je *typický* omyl? | smeru omylu |
| **GEH** | Sedí to aj na malých uliciach? | (pre denný model nepoužiteľné) |
| **screenline** | Sedí to na *koridoroch*, nie len na linkoch? | detailom |

---

### R² (koeficient determinácie)

**Čo je to:** „Koľko z rozdielov medzi úsekmi model vysvetlí."

```
R² = 1 − Σ(M − O)² / Σ(O − Ō)²
```

Menovateľ je, ako veľmi sa reálne úseky líšia navzájom. Čitateľ je, ako veľmi sa
model mýli. Ak sa model mýli málo v porovnaní s prirodzeným rozptylom, R² je
blízko 1.

| Hodnota | Význam |
|---|---|
| 1.0 | dokonalá zhoda |
| 0.8 | model vysvetlí 80 % rozdielov medzi úsekmi |
| 0.0 | model je presne taký dobrý ako „všade napíš priemer" |
| záporné | model je **horší** než priemer |

**Prah:** ≥ 0.80 ([defaults.py:357](src/sim/defaults.py#L357))
**Váš holdout:** 0.700 ❌

**Na čo si dať pozor:** R² je slepé voči systematickému posunu. Keby model všade
predpovedal presne polovicu reality, R² by bolo stále vysoké — poradie úsekov by
sedelo dokonale. Preto sa nikdy nehodnotí samo.

---

### slope a intercept (regresná priamka)

Cez body `(observed, modeled)` sa preloží priamka:

```
M = slope × O + intercept
```

**slope** je sklon. Ideál 1.0 = model rastie s realitou v pomere 1:1.

| slope | Význam |
|---|---|
| 1.0 | ideál |
| 0.71 | model rastie **pomalšie** — čím väčší úsek, tým väčšie podhodnotenie |
| 1.3 | model preháňa rozdiely medzi úsekmi |

**Prah:** 0.85 – 1.15
**Váš holdout:** 0.715 ❌

**Čo to prakticky znamená:** slope 0.715 s interceptom −1218 hovorí, že na
frekventovaných úsekoch model chýba viac než na tichých. Typická príčina je, že
model nedostatočne koncentruje dopravu na hlavné ťahy — buď je v OD matici málo
dlhých ciest, alebo sú kapacity hlavných ťahov nastavené príliš nízko, takže
priraďovanie dopravu „rozptýli" do bočných ulíc.

---

### bias_pct (systematická odchýlka)

**Najjednoduchšia metrika:** ako sa líši *súčet* modelu od *súčtu* reality.

```
bias = (ΣM − ΣO) / ΣO × 100
```

| bias | Význam |
|---|---|
| 0 % | model má správne celkové množstvo dopravy |
| −18.9 % | model má **o 19 % menej** dopravy než realita |

**Prah:** |bias| ≤ 15 %
**Váš holdout:** −18.9 % ❌

**Ako sa to opravuje:** nedostatok objemu sa opravuje na strane dopytu — viac
ciest do OD matice. Presne preto je zaujímavý parameter
`external_local.total_daily_trips`, ktorý rozoberá
[KAMDOJIZDIME_PLAN.md](KAMDOJIZDIME_PLAN.md).

**Pozor:** bias je slepý voči rozdeleniu. Model môže mať bias 0 % a pritom mať
všetku dopravu na nesprávnych uliciach.

---

### RMSE a %RMSE (typická veľkosť omylu)

```
RMSE  = √( priemer z (M − O)² )
%RMSE = RMSE / priemer(O) × 100
```

RMSE je v vozidlách, %RMSE v percentách priemerného úseku. Kvôli umocneniu na
druhú **trestá veľké omyly neúmerne viac** než malé — jeden úsek s omylom 20 000
váži rovnako ako 400 úsekov s omylom 1 000.

| %RMSE | Význam |
|---|---|
| 0 % | dokonalé |
| 42.2 % | typický omyl je 42 % priemerného úseku |

**Prah:** ≤ 35 %
**Váš holdout:** 42.2 % ❌ (RMSE 6 583 vozidiel)

**Rozdiel oproti biasu:** bias je *smerový* (plus a mínus sa vyrušia), %RMSE je
*absolútny*. Model s biasom 0 % môže mať %RMSE 60 %, ak sa polovica úsekov mýli
o +50 % a druhá o −50 %.

---

### MAPE a wMAPE

**MAPE** = priemer z `|M − O| / O` × 100. Priemerná percentuálna chyba.

Problém: MAPE dáva **rovnakú váhu úseku s 500 a s 50 000 vozidlami**. Malá tichá
ulička s omylom 200 vozidiel (40 %) pokazí MAPE rovnako ako diaľnica s omylom
20 000 (40 %). Preto sa pridáva:

**wMAPE** (vážená) = `Σ|M − O| / ΣO` × 100 — omyly aj skutočnosť sa najprv sčítajú.
Veľké úseky teda vážia viac, čo je vecne správne.

**Prah wMAPE:** varovanie nad 47 %
([validation.py:1496](src/sim/calibration/validation.py#L1496))
**Vaše MAPE:** 36.5 %

---

### GEH — a prečo je vo vašom reporte nepoužiteľný

**GEH** je dopravná štatistika (Geoffrey E. Havers), navrhnutá práve preto, že
percentá pri malých číslach klamú:

```
GEH = √( 2 × (M − O)² / (M + O) )
```

Kúzlo je v menovateli — pri veľkých objemoch toleruje väčšie absolútne rozdiely:

| Realita | Model | Rozdiel | GEH |
|---|---|---|---|
| 100 | 130 | +30 % | 2.8 ✅ |
| 10 000 | 13 000 | +30 % | 28.0 ❌ |
| 10 000 | 10 500 | +5 % | 4.9 ✅ |

Štandard (FHWA, britský DMRB): **85 % úsekov má mať GEH < 5**.

**⚠️ Ale: GEH je definovaný pre HODINOVÉ toky.** Váš model počíta **denné** objemy,
ktoré sú ~10× väčšie. GEH rastie približne s odmocninou objemu, takže rovnaká
percentuálna presnosť dá pri dennom modeli ~3× vyšší GEH.

Preto:

- **`geh_lt5_pct: 0.0` vo vašom reporte nie je katastrofa** — je to očakávané.
- Kód počíta aj **upravený prah** `daily_geh_threshold = 5 × √K`, kde K je
  `daily_capacity_factor` ([metrics.py:92](src/sim/calibration/metrics.py#L92)).
  U vás `daily_geh_threshold = 15.8`, a `daily_geh_lt_adj_pct = 21.4 %`.
- **GEH je zámerne vylúčený z konvergenčných kritérií** pre denný model
  ([context.py:765](src/sim/calibration/context.py#L765)). Verdikt PASS/FAIL stojí
  na R², slope, %RMSE, bias a screenlinoch.

Varovanie `"geh_lt5_pct<7"` v `quality_gates` je preto informatívne, nie fatálne.

---

### Screenline metriky

Zatiaľ čo predchádzajúce metriky merajú **jednotlivé linky**, screenline meria
**celý koridor** — koľko áut prekročí pomyselnú čiaru naprieč mestom.

To je robustnejšie: ak model pošle dopravu vedľajšou paralelnou ulicou, na úrovni
linkov to vyzerá ako dve chyby, ale screenline to zachytí správne.

```
screenline_max_error_pct = najhorší koridor z 21 porovnávaných
```

**Prah:** ≤ 15 %
**Váš stav:** 58.4 % ❌ (21 porovnávaných, 0 vylúčených)

Screenline, ktoré vyjdú extrémne (pomer model/realita mimo 0.2–5.0), sa
**automaticky vylúčia** z hodnotenia — obvykle to znamená chybu v napárovaní, nie
chybu modelu. U vás sa nevylúčila žiadna, takže tá 58 % odchýlka je reálna.

---

### Spearman ρ (poradová korelácia)

Bežná (Pearsonova) korelácia meria lineárny vzťah. **Spearman** meria len
**poradie** — či model správne určí, ktoré ulice sú frekventovanejšie, bez ohľadu
na absolútne čísla.

Je to užitočné tam, kde sa porovnávajú **nesúmerné veličiny** — napr. modelový
objem vozidiel proti Waze zdržaniu v sekundách. Absolútne čísla porovnať nejde,
poradie áno.

---

### Holdout vs. kalibračná časť — najdôležitejší rozdiel v celom reporte

Report ukazuje **dve sady tých istých metrík**:

| | `calibration_reference` | `holdout_validation` |
|---|---|---|
| dáta | 65 % CSD úsekov | 35 % CSD úsekov |
| ODME ich videla? | **áno** | **nie** |
| R² u vás | 0.839 | **0.700** |
| slope | 0.932 | **0.715** |
| %RMSE | 38.9 | **42.2** |
| bias | −14.0 % | **−18.9 %** |

**Verdikt sa berie z holdoutu** (`verdict_source: "holdout"`), a to je správne.
Kalibračné číslo je vždy optimistickejšie — ODME sa na tie dáta priamo fitovala.
Rozdiel medzi 0.839 a 0.700 je presne miera toho, koľko z „úspechu" je fit
a koľko skutočná schopnosť modelu.

**`holdout_adequacy: "thin"`** = holdout má len 14 použiteľných bodov. Pri takom
počte je jedno-dve odľahlé merania schopné pohnúť R² o desatiny. Preto pri
porovnávaní dvoch verzií modelu vždy uvádzajte aj `n` a neinterpretujte malé
rozdiely.

---

### Prečo je použiteľných len 14 z desiatok CSD úsekov

`csd_summary` ukazuje, koľko sa ich cestou vyradilo:

```
n_roads: 8
n_partial_excluded:         6   ← model pokrýva < 70 % dĺžky CSD úseku
n_minimal_excluded:         3   ← pokrýva < 30 %
n_over_aggregated_excluded: 2   ← model má > 2× viac km než CSD úsek
n_zero_flow_excluded:       1   ← model tam predpovedal nulu
n_ratio_excluded:           1   ← pomer mimo pásma 0.2–5.0
```

Toto **nie sú chyby modelu, ale chyby napárovania.** CSD úsek „I/42 od km 12.3 po
14.1" nemusí zodpovedať tomu, čo model považuje za tú istú cestu. Porovnávať
priemer 40 modelových linkov s priemerom celkom iného rozsahu by dalo nezmyselné
číslo, tak sa to radšej vyradí.

Dôsledok: **zvýšenie počtu použiteľných bodov je samo o sebe zlepšenie modelu** —
aj keby sa metriky nezmenili. Robí sa to ladením `matching` parametrov
v [config/brno/sim.yaml](config/brno/sim.yaml).

---

### Quality gates — mäkké a tvrdé

```json
"quality_gates": {
  "hard_class_bias_max_abs_pct": 90.0,
  "warn_wmape_pct": 47.0,
  "warn_geh_lt5_pct": 7.0,
  "hard_reject": false,
  "warnings": ["geh_lt5_pct<7"]
}
```

- **`hard_reject`** = model je nepoužiteľný. Spustí sa len ak niektorá trieda ciest
  má odchýlku > 90 % — teda napr. „na všetkých diaľniciach je model dvojnásobne
  vedľa". U vás `false` ✅
- **`warnings`** = stojí za pozretie, ale nie je to blokujúce.

---

### Zhrnutie vášho aktuálneho stavu

| Metrika | Holdout | Cieľ | |
|---|---|---|---|
| R² | 0.700 | ≥ 0.80 | ❌ |
| slope | 0.715 | 0.85–1.15 | ❌ |
| %RMSE | 42.2 | ≤ 35 | ❌ |
| bias | −18.9 % | ≤ 15 % | ❌ |
| screenline max | 58.4 % | ≤ 15 % | ❌ |
| hard_reject | false | false | ✅ |

**Čo to hovorí dohromady:** model **systematicky podhodnocuje dopravu** (bias −19 %)
a to podhodnotenie **rastie s veľkosťou úseku** (slope 0.715). To je konzistentný
obraz, nie náhodný šum — a znamená to, že chyba je s najväčšou pravdepodobnosťou
v **objeme a štruktúre OD matice**, nie v priraďovaní.

Presne to je dôvod, prečo majú kamdojizdime dáta zmysel: dávajú nezávislý odhad
externých objemov namiesto dnešného hádaného čísla.

---

## Scenáre — čo model reálne umožňuje

[src/sim/scenarios/engine.py](src/sim/scenarios/engine.py)

Toto je pointa celého projektu. Cez API pošlete: *„zavri link 12345"* alebo
*„zredukuj z 3 pruhov na 1"*.

Kód potom:

1. Načíta skalibrovaný graf
2. **V pamäti** ([engine.py:43](src/sim/scenarios/engine.py#L43)) nastaví
   zavretému linku kapacitu 0.001 a čas 99 999 s — BPR ho tým urobí prohibitívne
   drahým, takže nikto ním nepôjde. Pri redukcii pruhov len škáluje kapacitu.
3. Spustí nové priradenie
4. Porovná s baseline a spočíta rozdiely: `delta_vol`, `delta_pct`, `delta_voc`,
   `significance`

Výsledok je mapa, kde vidíte, **kam sa doprava presunie**, keď zavriete konkrétnu
ulicu. Beží to v background threade, frontend polluje stav.

---

## Ako je repozitár organizovaný

```
run.py                      ← rozcestník všetkých krokov
src/sim/
  defaults.py               ← VŠETKY defaultné hodnoty (717 riadkov)
  io_project.py             ← načítanie configu, cesty per mesto
  api.py                    ← FastAPI server
  network/                  ← import OSM + normalizácia (~2500 r.)
  zoning/                   ← zóny, brány, konektory (~2900 r.)
  datasets/                 ← sťahovanie dát (~2400 r.)
  supernetwork/             ← národná sieť pre tranzit (~1500 r.)
  demand/                   ← OD matica (~2000 r.)
  distribution/             ← gravitácia + IPF (~700 r.)
  assignment/               ← ekvilibrium (~1000 r.)
  calibration/              ← ODME + validácia (~8500 r., najväčší modul)
  scenarios/                ← what-if analýzy
config/<mesto>/sim.yaml     ← len odchýlky od defaultov
experiments/                ← exp01–exp11, podklady pre diplomovku
```

**Návrhový princíp, ktorý sa v projekte drží:** metodologické defaulty žijú
v [defaults.py](src/sim/defaults.py), configy miest obsahujú len to, čo je pre
dané mesto špecifické. Config Brna má **60 riadkov** — všetko ostatné sa dedí.
Vďaka tomu pridanie mesta znamená vygenerovať krátky YAML
(`python run.py init-city`).

---

## Čo je na tom v skutočnosti ťažké

Aby ste mali predstavu, kde je skutočná práca (a prečo má calibration modul
8 500 riadkov):

1. **Napárovanie meraní na sieť.** ŘSD má sčítací úsek „I/42, od km 12.3 po
   km 14.1". OSM má 40 linkov. Ktoré patria k sebe? Rieši sa geometrickými
   buffermi, kontrolou čísla cesty, agregáciou koridoru, skóre kvality zhody, a
   detekciou rozdelených jazdných pásov.
   [matching.py](src/sim/calibration/matching.py)

2. **Deravé OSM dáta.** Chýbajúce rýchlosti, chýbajúce pruhy,
   `highway=construction` namiesto skutočnej triedy, umelé jednopruhové hrdlá.
   Tri samostatné heuristiky v
   [normalization.py](src/sim/network/normalization.py) sa snažia toto opravovať.

3. **Nedourčenosť ODME.** Máte pár stoviek meraní a desiatky tisíc buniek matice.
   Matematicky má úloha nekonečne veľa riešení. Preto tie zábrany: hranice
   `max_deviation`, damping, revert, penalizácia odchýlky od pôvodného odhadu.

4. **Denný vs. hodinový model.** Väčšina literatúry a metrík (GEH, kapacity, BPR)
   predpokladá špičkovú hodinu. Tento model beží denne, takže sa všade
   prepočítava — `daily_capacity_factor`, upravený GEH prah `5×√K`, K-faktory na
   špičku.

5. **Tranzit.** Celý supernetwork modul existuje len preto, aby sa dala odhadnúť
   doprava, ktorá v Brne nič nemá, ale prechádza cezeň.
