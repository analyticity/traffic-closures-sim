# Pipeline krok za krokom — technický rozbor

> Doplnok k [popis_simulacie.md](popis_simulacie.md). Zatiaľ čo `popis_simulacie.md`
> vysvetľuje model laikovi, tento dokument ide do kódu: **čo do každého kroku
> vstupuje, aký algoritmus tam beží, kde sú riziká, kde sú konkrétne chyby a čo
> by sa dalo zlepšiť.**
>
> Písané pre obhajobu — čokoľvek tu označené ako riziko je lepšie povedať samej
> než počuť od oponenta.

---

## Ako čítať tento dokument

Každý krok má rovnakú štruktúru:

- **Vstupy** — čo musí existovať, aby krok bežal
- **Algoritmus** — čo sa reálne deje, s odkazmi do kódu
- **Výstupy** — čo po ňom zostane na disku
- **Rizikové miesta** — metodické slabiny, nie chyby v kóde
- **Nálezy v kóde** — konkrétne veci, ktoré vyzerajú ako chyba alebo latentný problém
- **Čo zlepšiť** — konkrétne, zoradené podľa pomeru prínos/práca

Nálezy sú označené:
**[M]** metodický problém · **[B]** pravdepodobná chyba ·
**[L]** latentný problém (dnes nepadá, ale môže) · **[D]** rozpor s dokumentáciou

---

## Zhrnutie: 10 najdôležitejších nálezov

| # | Kde | Nález | Typ |
|---|---|---|---|
| 1 | `distribute` | Segment `other` sa generuje dvakrát dvomi rôznymi sadami parametrov; IPF ten prvý ticho prepíše (+71 000 vozidiel/deň) | **M** |
| 2 | `calibrate` | Sčítania, ktoré model netrafí (pomer mimo [0.2, 5]) sa vylúčia z kalibrácie **aj z metrík** | **M** |
| 3 | `validate` | „Corridor" split nie je náhodný — najväčšie cesty idú do kalibrácie, holdout dostáva zvyšky | **M** |
| 4 | `calibrate` | Globálny reziduál škáluje celú maticu (461 tis. ciest) podľa pomeru nameraného na ~14 úsekoch | **M** |
| 5 | `fetch-data` | Nespárovaná zóna dostane **medián** populácie všetkých obcí ČR | **B** |
| 6 | `fetch-data` | Zamestnanosť = počet dochádzajúcich zvonku → obce, kde ľudia pracujú doma, majú atrakciu ~0 | **M** |
| 7 | `normalize-network` | Kapacita sa odvodzuje z CSD AADT, proti ktorému sa potom model kalibruje (cirkularita) | **M** |
| 8 | `normalize-network` | `geometry.length` pri chýbajúcej dĺžke vráti stupne, nie metre | **L** |
| 9 | `distribute` | Zlyhanie AequilibraE gravitácie/IPF sa ticho prehltne a nasadí sa hrubší vlastný fallback | **B** |
| 10 | `calibrate` | To, čo kód volá „Spiess", je heuristická proporcionálna aktualizácia, nie Spiessov gradient s line-search | **D** |

---

## Stav nálezov k 28. 8. 2026

> Dokument bol napísaný 18. 8. 2026. Odvtedy prišlo ~20 commitov. Táto tabuľka
> hovorí, čo z neho ešte platí. Referenčný beh je
> [`updated_version_6`](simulation_for_article/updated_version_6/) (commit `1f19c7a`).

| # | Nález | Stav | Poznámka |
|---|---|---|---|
| 1 | `other` sa generuje dvakrát | 🔴 **platí** | `defaults.py:231` (1.0/0.35/1.50) vs `defaults.py:288` (1.8/0.40/1.3) — nezmenené |
| 2 | Vylúčenie podľa pomeru mimo [0.2, 5] aj z metrík | 🟠 **čiastočne** | Ručné `exclude_csd_roads` a `csd_validation_exclude_sil` sú prázdne (n vyskočilo z 14 na 24). Automatické vylúčenie podľa pomeru v `matching.py` je **stále jeden flag `_excluded`**, nerozdelený na geom/objem |
| 3 | „Corridor" split nie je náhodný | 🔴 **platí** | `observed.py:555` stále `sort_values("n_sections", ascending=False)` |
| 4 | Globálny reziduál škáluje celú maticu podľa ~14 úsekov | 🟠 **zmiernené** | Už 24 úsekov, ale mechanizmus rovnaký. `global_residual_damping` znížený na 0.10 (config) |
| 5 | Nespárovaná zóna dostane medián populácie ČR | 🔴 **platí** | `population.py:215` `default_median` |
| 6 | Zamestnanosť = len dochádzajúci zvonku | 🟠 **čiastočne** | Filter `lokalizace=0` stále aktívny (`employment.py:212`), ale pribudol `city_split_weights` — rozdelenie zamestnanosti Brna na 29 MČ podľa OSM pracovísk namiesto podľa populácie (centrum 42,1 % namiesto 17,8 %) |
| 7 | Kapacita z CSD AADT → cirkularita | 🔴 **platí** | nezmenené |
| 8 | `geometry.length` vracia stupne | 🔴 **platí** | `normalization.py:722` nezmenené |
| 9 | Tiché fallbacky v gravitácii/IPF | 🔴 **platí** | `gravity.py:46`, `gravity.py:93` — stále `except Exception` |
| 10 | „Spiess" nie je Spiess | 🔴 **platí, a je to horšie** | viď nižšie |

### Nález 10 sa medzitým zdvojil

Referenčný beh v6 má v `calibration_report.json` `"method": "entropy_odme"` —
teda **nebežal Spiess vôbec**. Default v [defaults.py:314](src/sim/defaults.py#L314)
je `entropy_odme`, zatiaľ čo docstring v [run.py:18](run.py#L18) tvrdí
„Spiess gradient ODME (default)" a oba popisné dokumenty opisujú Spiessa.

Do práce aj do článku teda treba:

1. opísať **entropy-maximization** vetvu (`entropy_step_size = 0.15` v Brne), nie Spiessa;
2. buď zosúladiť docstring `run.py`, alebo default prepnúť späť;
3. ak sa Spiess spomína, tak ako *alternatívna* implementovaná metóda.

### Čo pribudlo a v dokumente ešte nie je rozobrané

| Zmena | Kde | Dopad |
|---|---|---|
| 8 brán namiesto 5 (I/50, 602, 380, 430) | `config/brno/sim.yaml:96-99` | pokrylo ~47 % predtým chýbajúcej hraničnej dopravy |
| Kotva brány I/43 južne od zúženia | `sim.yaml:85` | modelovaných 15 118 → dopyt cez bránu prejde |
| 29 mestských častí z Overpassu namiesto osmnx | `sim.yaml:53` + `zones_admin9.geojson` | doplnené Brno-střed, Komín, Chrlice (osmnx ich ticho vynechával) |
| `external_local` 110 000 + `corridor_weights` z kamdojizdime | `sim.yaml:135-167` | holdout R² 0.466 → 0.685 |
| `city_split_weights` (zamestnanosť z OSM POI) | `sim.yaml:238` | holdout slope 0.715 → 0.794 |
| Zrušené ručné vylúčenia CSD | `sim.yaml:202-203` | n z 14 na 24 |
| Null scenár (zavretie linky s nulovým objemom) | notebook | odhalil šumové dno priradenia — viď nižšie |

### Nový nález: šumové dno scenárov **[M]**

V behu v6 sa zavrela jedna linka s nulovým objemom. Výsledok: **753 hrán**
zmenilo objem o viac než 500 voz/deň, Σ|Δ| = **1,30 mil. voz/deň** (2,5 %
celkového objemu siete), ΔVHT = **−1 424 voz·h**. Reálny scenár (Jihlavská,
75 liniek, 414 tis. voz/deň) dá Σ|Δ| = 4,64 mil. a ΔVHT = **−1 079**.

Dôsledky:

- **Δ objemov je použiteľné** (signál/šum ≈ 3,6), ale každé scenárové číslo sa
  musí uvádzať proti null scenáru.
- **ΔVHT a ΔVKT sú dnes pod šumovým dnom** a nesmú sa publikovať tak, ako sú.
  Príčina je tolerancia BFW (`rgap_target` 0.002 pri baseline,
  `scenario_rgap` pri scenári) — dve nezávislé aproximácie toho istého
  ekvilibria sa odčítavajú.
- Oprava je lacná: baseline aj scenár spustiť s rovnakým, prísnym `rgap`
  (1e-4) a z rovnakého štartu, alebo KPI počítať len na linkách v okolí zásahu.
  Detail v [plan_vyhodnotenia.md](plan_vyhodnotenia.md), krok E1.

---

## 0. Orchestrácia — `run.py`

**Vstupy:** `config/<mesto>/sim.yaml` (deep-merge nad `SIM_DEFAULTS`
v [defaults.py](src/sim/defaults.py), viď [io_project.py:96](src/sim/io_project.py#L96)).

**Algoritmus:** `run.py` nie je DAG runner — je to zoznam `STEPS`
([run.py:79](run.py#L79)) a dve deklaratívne tabuľky:

- `_STEP_PREREQUISITES` ([run.py:156](run.py#L156)) — kontrola existencie súborov
  pred spustením kroku (projekt DB, matica, výsledky priradenia, zóny)
- `_STALENESS_CHECKS` ([run.py:199](run.py#L199)) — porovnanie `mtime` dvojíc súborov,
  varuje (nezastaví), keď je výstup starší než vstup

**Rizikové miesta**

- **Staleness sa iba loguje ako WARNING.** Keď zmeníš config a spustíš len
  `calibrate`, model sa odkalibruje na starej sieti a nikto to nezastaví.
  Staleness pokrýva navyše len 4 kroky zo 17.
- **Žiadny hash configu.** Staleness sa pozerá na `mtime` súborov, nie na obsah
  konfigurácie. Zmena `through_traffic_scale` v YAML neurobí nič „staré",
  hoci mení výstup zásadne. (Výnimka: kalibrácia si hashuje vstupy interne —
  `_file_hash` v [context.py:322](src/sim/calibration/context.py#L322).)
- **`clean` maže `outputs/<mesto>/`, `data/<mesto>/` aj celý AequilibraE projekt.**
  Bez `--force` prežije len `data/sources/`. Nie je tam žiadne potvrdenie.

**Čo zlepšiť**

1. Zapísať hash relevantnej časti configu do každého výstupného JSON-u a
   porovnávať ho v `_check_step_prerequisites` — jednoduché, veľký prínos pre
   reprodukovateľnosť experimentov.
2. Prepnúť staleness z WARNING na chybu s `--allow-stale` únikom.

---

## 1. `build-network` — import cestnej siete z OSM

**Vstupy:** `osm.place_name` alebo `model_bbox`, `osm.buffer_km`, sieťové filtre.

**Algoritmus** ([network/pipeline.py:386](src/sim/network/pipeline.py#L386))

1. **Rozlíšenie územia** — `geocode_place` (osmnx/Nominatim) → polygón obce,
   buffer `buffer_km` + **5 km konektivitná rezerva**
   ([pipeline.py:46](src/sim/network/pipeline.py#L46)). Buffer sa robí cez metrický
   CRS (`buffer_polygon_km`), nie v stupňoch — správne.
2. **Import** — `project.network.create_from_osm(...)` (AequilibraE si sama
   ťahá dáta z Overpass a rozseká ways na linky).
3. **Filtre** ([pipeline.py:173](src/sim/network/pipeline.py#L173)):
   - *drivable* — vyhodí `footway/path/cycleway/steps/...` a všetko bez módu `c`
   - *isolated components* — `networkx.connected_components` na **neorientovanom**
     multigrafe, ponechá najväčší komponent podľa počtu hrán
     ([filtering.py:98](src/sim/network/filtering.py#L98))
4. **Orez na mestské jadro** ([pipeline.py:214](src/sim/network/pipeline.py#L214)) —
   `compute_urban_trim_bbox` ([crs.py:121](src/sim/network/crs.py#L121)) vezme uzly,
   ktoré sa dotýkajú aspoň jednej **ne-koridorovej** cesty (tj. nie
   motorway/trunk/primary), odreže 1 % kvantil z každej strany a pridá 2 % padding.
   Koridorové linky sa **nikdy neorezávajú** — musia dosiahnuť až k bránam.
5. **Obohatenie z OSM** ([osm_enrichment.py:361](src/sim/network/osm_enrichment.py#L361)) —
   druhý, nezávislý download cez `osmnx` (`network_type="drive"`,
   `simplify=False`), agregácia tagov `ref`/`name`/`highway` podľa `osmid`
   (najčastejšia hodnota) a join na linky cez `osm_id`.
   Druhý priechod `fill_missing_refs_from_named_corridors`
   ([osm_enrichment.py:243](src/sim/network/osm_enrichment.py#L243)) dopĺňa chýbajúce
   `ref` pozdĺž koridorov s rovnakým názvom (DFS po komponente, dominantný ref
   musí mať ≥ 80 % podiel).

**Výstupy:** `project/<mesto>_aeq/` (SQLite + SpatiaLite), `network_counts.json`,
`links_native.geojson`, mapy PNG.

**Rizikové miesta**

- **Dva rôzne OSM snapshoty v jednom kroku.** AequilibraE si sťahuje sieť sama,
  `osmnx` sťahuje druhýkrát pre obohatenie. Medzi tými dvoma requestmi môže byť
  iný stav OSM a hlavne **iná filtračná logika** — `osmnx drive` vynecháva napr.
  `highway=construction`, takže tie linky zostanú s prázdnym `osm_highway`
  (kód to potom rieši „corridor promotion", viď krok 3).
- **Konektivita sa rieši neorientovane.** Najväčší komponent podľa
  neorientovaného grafu môže obsahovať jednosmerné pasce (uzol, do ktorého sa dá
  vojsť, ale nedá vyjsť). Orientovaná kontrola prichádza až v kroku 3.
- **Reprodukovateľnosť:** OSM sa mení denne, `place_name` geokóduje živý
  Nominatim. Ten istý príkaz o pol roka postaví inú sieť.

**Nálezy v kóde**

- **[L]** `trim_network_to_bbox_raw` maže uzly jedným SQL-om
  s `IN (?,?,...)` bez chunkovania
  ([filtering.py:216-218](src/sim/network/filtering.py#L216-L218)), hoci
  `bulk_delete_by_ids` s chunkom 450 existuje hneď vedľa
  ([db.py:52](src/sim/network/db.py#L52)). Pri starších SQLite (limit 999
  premenných) to spadne; na dnešných (32 766) prejde. Rovnaká vec sa zopakuje
  v normalizácii (krok 3).
- **[L]** Ten istý `trim_network_to_bbox_raw` maže linky po jednej cez
  API v `try/except: pass` ([filtering.py:208-213](src/sim/network/filtering.py#L208-L213))
  — tiché zlyhanie a nafúknutý čas behu (desaťtisíce volaní).
- **[B]** `enrich_links_from_osm` najprv **nastaví všetky `osm_ref`,
  `osm_ref_norm`, `osm_name_raw`, `osm_highway` na NULL**
  ([osm_enrichment.py:456-459](src/sim/network/osm_enrichment.py#L456-L459)) a až
  potom zapisuje nové hodnoty. Keď sa `osmnx` download nepodarí čiastočne
  (menšia bbox, výpadok Overpassu), tichý výsledok je sieť bez referencií ciest
  — a tá istá sieť potom nedokáže párovať CSD sčítania. Malo by sa zapisovať do
  dočasného stĺpca a prepnúť až po úspechu.
- **[L]** `guess_crs_from_coords` ([crs.py:20](src/sim/network/crs.py#L20)) háda CRS
  podľa toho, či súradnice vyzerajú ako stupne. Pre ČR to funguje, ale je to
  heuristika v ceste, kde stačí explicitná hodnota.

**Čo zlepšiť**

1. Zapísať do `network_counts.json` timestamp a bbox oboch downloadov + počet
   liniek bez `osm_ref` (jednoduchý ukazovateľ kvality obohatenia).
2. Nahradiť mazanie po jednom za `bulk_delete_by_ids` — rýchlosť aj korektnosť.
3. Zvážiť použitie jedného zdroja OSM (napr. lokálny `.osm.pbf` snapshot),
   čím sa krok stane reprodukovateľným a odpadne druhý download.

---

## 2. `fetch-data` — externé dáta

**Vstupy:** `datasets.sources.*` (URL, cesty), voliteľne Postgres pre uzávierky
a Waze.

**Algoritmus** ([datasets/pipeline.py](src/sim/datasets/pipeline.py)) — sťahovanie
+ preprocessing štyroch rodín dát:

| Dáta | Zdroj | Spracovanie |
|---|---|---|
| SLDB dochádzka | ČSÚ 2021 | CSV → parquet, filter podľa okresu/ORP |
| SLDB populácia | ČSÚ 2021 | CSV → `zone_population.parquet` (matching na zóny) |
| Zamestnanosť | odvodená z dochádzky | `zone_employment.parquet` |
| CSD 2025 | ŘSD XLSX | auto-detekcia hárku a hlavičky, normalizácia stĺpcov |
| Uzávierky / Waze | NDIC, Postgres | geometrie + časové okná |

**CSD parsing** ([datasets/csd.py:127](src/sim/datasets/csd.py#L127)) skenuje prvé
4 riadky každého hárku a vyberie ten s najviac známymi stĺpcami
(`sil`, `rpdi`, `sv`, `o`, `tv`, ...). `normalize_csd_count_columns`
([csd.py:47](src/sim/datasets/csd.py#L47)) dopočíta `sv`/`o`/`tv`, keď export
publikuje len triedy vozidiel.

**Populácia** ([datasets/population.py:60](src/sim/datasets/population.py#L60)) —
kaskáda párovania názvu zóny na obec/mestskú časť:

1. zóna patrí do mestskej časti (mapovanie `locale.yaml`) → **podiel podľa plochy**
2. názov zóny = názov mestskej časti (`uzemi_cis=44`) → celá populácia MČ
3. presná zhoda s obcou (`uzemi_cis=43`)
4. zhoda po odstránení prípony (`-mesto`, ` u Brna`, ...)
5. fuzzy `difflib` s prahom 0.82
6. **fallback: medián populácie všetkých obcí**

**Rizikové miesta**

- **Zamestnanosť je proxy, nie dáta.** `derive_zone_employment`
  ([employment.py:36](src/sim/datasets/employment.py#L36)) definuje zamestnanosť
  zóny ako **počet ľudí, ktorí do nej dochádzajú zvonku**, pričom riadky
  `lokalizace = 0_na_adrese_OP` (pracujúci v mieste bydliska) sa **explicitne
  vyhadzujú** ([employment.py:96-99](src/sim/datasets/employment.py#L96-L99)).
  Obec, kde väčšina ľudí pracuje doma, teda vyjde ako miesto bez pracovných
  miest. Keďže atrakcie v `distribute` sú priamo úmerné tomuto číslu, chyba sa
  prenáša do priestorového rozloženia ciest.
- **Nespárované zóny majú zamestnanosť 0**
  ([employment.py:183-187](src/sim/datasets/employment.py#L183-L187)) — v IPF to
  znamená „sem nikto nejde", čo je silnejšie tvrdenie než „nevieme".
- **CSD 2025 vs SLDB 2021** — sčítanie dopravy a sčítanie ľudu sú 4 roky od seba,
  a rok 2021 bol covidový. Nikde sa to nekompenzuje explicitne.

**Nálezy v kóde**

- **[B]** `default_median` ([population.py:213-215](src/sim/datasets/population.py#L213-L215)):
  zóna, ktorá neprejde ani fuzzy zhodou, dostane **medián populácie všetkých
  obcí a mestských častí** v datasete. To nie je konzervatívny odhad — je to
  vymyslené číslo, ktoré ide priamo do produkcie ciest v gravitačnom modeli.
  Lepšie: 0 + hlasný WARNING + zápis do reportu.
- **[B]** Zamietnutá fuzzy zhoda končí **v tom istom mediánovom fallbacku**
  ([population.py:203-215](src/sim/datasets/population.py#L203-L215)). Kód správne
  rozpozná „táto zhoda je podozrivá, obec má > 5000 obyvateľov" a potom miesto
  0 priradí medián. Zamestnanosť to rieši správne (`fuzzy_rejected` → 0,
  [employment.py:173-176](src/sim/datasets/employment.py#L173-L176)) — populácia
  by mala robiť to isté.
- **[M]** `max_zone_mc_share = 0.70` ([population.py:157](src/sim/datasets/population.py#L157)):
  keď je mestská časť rozdelená na katastre, každý dostane svoj podiel plochy,
  ale zastropovaný na 70 %. Podiely sa **nerenormalizujú**, takže keď jeden
  kataster tvorí 90 % plochy MČ, 20 % populácie sa jednoducho stratí. Celková
  populácia modelu potom nesedí na SLDB.
- **[M]** Deduplikácia obcí `sort_values("hodnota", ascending=False).drop_duplicates("nazev")`
  ([population.py:117](src/sim/datasets/population.py#L117)): v ČR je mnoho
  rovnomenných obcí (Lhota, Nová Ves...). Kód si vždy vyberie tú najväčšiu, čím
  systematicky nadhodnocuje. Správne by bolo párovať cez `uzemi_kod`
  (číselný kód obce), ktorý v dátach je a načítava sa, ale nepoužíva sa na join.
- **[L]** Odstránenie prípony je `znorm.replace(suffix, "")`
  ([population.py:189](src/sim/datasets/population.py#L189)) — nahrádza podreťazec
  kdekoľvek, nie len na konci.

**Čo zlepšiť**

1. Párovať zóny na obce cez **kód obce**, nie cez názov. Kód je v CSV
   (`uzemi_kod`) aj v OSM (`ref:nuts`, `ref:LAU`) — odpadne celá fuzzy kaskáda
   aj mediánový fallback.
2. Vyhodiť `default_median`, nahradiť 0 + záznam v `zone_population.parquet`
   stĺpci `match`, a v `build-demand` hlásiť, koľko % populácie modelu je
   nespárovaných.
3. Zamestnanosť dopočítať z celkovej dochádzky **vrátane** `lokalizace=0`
   (pracujúci v mieste), inak sú atrakcie štrukturálne vychýlené v prospech
   centra.

---

## 3. `normalize-network` — rýchlosti, kapacity, BPR, uzávierky

Toto je najhustejší krok v repozitári ([normalization.py](src/sim/network/normalization.py),
948 riadkov) a zároveň ten, kde sa dá najviac pokaziť potichu.

**Vstupy:** AequilibraE projekt, `network_normalization.yaml`, voliteľne CSD parquet.

**Algoritmus** ([normalization.py:483](src/sim/network/normalization.py#L483)), v poradí:

1. **Obnova typov liniek** — AequilibraE zlučuje `*_link` na rodičovský typ;
   kód ho vracia späť z `osm_highway` ([normalization.py:518-529](src/sim/network/normalization.py#L518-L529)).
2. **Reklasifikácia `construction`** — podľa rýchlosti a smeru na `motorway_link`
   / `trunk_link` / `residential`.
3. **„Corridor promotion"** ([normalization.py:568-626](src/sim/network/normalization.py#L568-L626)) —
   DFS po linkách s rovnakým názvom: pomenované `residential`/`service` linky
   bez `osm_highway`, ktoré susedia s vyššou triedou toho istého názvu, sa
   povýšia na tú triedu. Rieši dieru po `highway=construction`.
4. **Doplnenie chýbajúcich atribútov** (`_resolve_directional`,
   [normalization.py:725](src/sim/network/normalization.py#L725)) — pre obojsmerné
   linky sa chýbajúci smer doplní z opačného, potom z defaultov podľa typu,
   nakoniec globálnym fallbackom. Každé doplnenie sa značí do `estimated_*`.
5. **Vynútenie minimálnej rýchlosti** — linka pod 40 % defaultu svojho typu sa
   zdvihne na default ([normalization.py:796](src/sim/network/normalization.py#L796)).
6. **Oprava „pinch pointov"** ([normalization.py:152](src/sim/network/normalization.py#L152)) —
   jednopruhové prepojky medzi obojsmernou cestou a rozdelenou diaľnicou sa
   dorovnajú na počet pruhov susednej linky (iteruje max. 5×).
7. **Praktická rýchlosť (HCM)** ([normalization.py:282](src/sim/network/normalization.py#L282)):
   ```
   practical = posted × base_factor[typ] − penalty[typ] × ipkm
   ```
   `ipkm` = počet koncových uzlov linky so stupňom ≥ 3 na kilometer.
   Idempotencia je zaistená stĺpcami `posted_speed_ab/ba`, ktoré si držia
   pôvodnú hodnotu.
8. **Kapacita** = `capacity_per_lane[typ] × lanes`, ak nie je z OSM.
9. **CSD capacity hints** ([normalization.py:83](src/sim/network/normalization.py#L83)) —
   kapacita sa zdvihne aspoň na `AADT × 0.10` na cestách spárovaných cez `ref`.
10. **Experiment profil** — stropy/podlahy rýchlosti, násobiče kapacity,
    časové penalizácie, potom globálne clip na `min_speed_kmh` / `min_capacity_vph`.
11. **Voľný čas jazdy** `t = distance × 3.6 / speed`, vynulovanie opačného smeru
    na jednosmerkách ([normalization.py:419](src/sim/network/normalization.py#L419)).
12. **Konektivitné opravy** ([connectivity.py](src/sim/network/connectivity.py)):
    - `repair_boundary_scc` ([connectivity.py:34](src/sim/network/connectivity.py#L34)) —
      jednosmerné `motorway_link`/`trunk_link`/`trunk` s koncom mimo najväčšej
      **orientovanej** silne súvislej komponenty sa spravia obojsmernými
    - `repair_divided_highway_dead_ends` ([connectivity.py:139](src/sim/network/connectivity.py#L139)) —
      slepé konce rozdelených jazdných pásov sa spoja krátkou linkou
      (2 priechody: najprv podľa zhodného `ref`, potom podľa typu a vzdialenosti;
      nová linka má 20 km/h a 200 voz/h, aby po nej model nechodil zbytočne)
13. **Uzávierky** ([closures.py](src/sim/network/closures.py)) — matchovanie podľa
    line geometrie (buffer nad `sindex`) alebo bodu, redukcia kapacity a
    rýchlosti podľa `severity_map`, so zálohou v `_preclosure_*` stĺpcoch.

**Rizikové miesta**

- **Väčšina siete je odhad, nie dáta.** `supply_audit` (krok 10) varuje, keď
  je > 70 % liniek imputovaných — v praxi je OSM `maxspeed` a `lanes` na
  mestských uliciach zriedkavý, takže rýchlosti a kapacity sú z prevažnej časti
  tabuľkové hodnoty podľa typu cesty.
- **`ipkm` je hrubá proxy.** Stupeň uzla ≥ 3 znamená „nejaká odbočka", nie
  „svetelná križovatka". Kruháče, prednosti a signalizácia sa nerozlišujú, takže
  penalizácia je rovnaká pre okresku s poľnou cestou aj pre mestský bulvár.
- **[M] Cirkularita CSD → kapacita → kalibrácia na CSD.** Krok 9 nastaví
  kapacitu ≥ `AADT × 0.10` (hodinovú). V priradení sa kapacita ešte násobí
  denným faktorom 10–13 ([graph.py:71-97](src/sim/assignment/graph.py#L71-L97)),
  takže denná kapacita takej linky je **≥ AADT**. Na spárovaných cestách teda
  V/C ≤ 1 už z konštrukcie a BPR tam prakticky nespomaľuje. Model sa potom
  kalibruje proti tomu istému AADT. Netvrdím, že je to nepoužiteľné — ale je to
  informácia z pozorovaní vpustená do ponukovej strany, presne to, čo `audit-supply`
  má strážiť ([context.py:72](src/sim/calibration/context.py#L72) hovorí
  „ODME nesmie kompenzovať chyby siete").
- **Obojsmerný `trunk`.** `_SCC_REPAIR_ELIGIBLE` zahŕňa mainline `trunk`
  ([connectivity.py:27](src/sim/network/connectivity.py#L27)). V ČR sú I. triedy
  často rozdelené štvorpruhy — spraviť takú linku obojsmernou znamená pustiť
  autá do protismeru. Kód to komentárom priznáva a robí to len na okraji modelu,
  ale je to presne tá vec, čo skresľuje bránové screenline.

**Nálezy v kóde**

- **[L]** Chýbajúca dĺžka sa dopĺňa ako `links.geometry.length`
  ([normalization.py:722](src/sim/network/normalization.py#L722)). Geometria
  v AequilibraE projekte je vo **WGS-84**, takže `.length` vráti **stupne**
  (~0.009 namiesto ~1000 m). Dnes to nevybuchne, lebo AequilibraE `distance`
  dopočíta pri importe, ale akonáhle vznikne linka bez dĺžky, dostane
  free-flow čas ~0 a stane sa z nej diaľkový skrat cez celý model.
  Oprava je jednoriadková: prepočítať cez metrický CRS.
- **[L]** `_apply_csd_capacity_hints` robí `merged = links[["_ref_norm"]].merge(csd_agg, ...)`
  a potom indexuje pôvodný `links` indexmi z `merged`
  ([normalization.py:120-137](src/sim/network/normalization.py#L120-L137)).
  `merge` vracia nový `RangeIndex` — pokiaľ `links` nemá rovnaký súvislý index
  (a nemá ho, keď sa predtým mazali riadky cez `links[~mask]`), kapacity sa
  zapíšu **nesprávnym linkám**. Dnes to prejde len preto, že mazanie
  (`link_removals`, `crossing`) beží *pred* touto funkciou a používa `.copy()`
  bez `reset_index()` — čiže index **je** nesúvislý, ak sa niečo zmazalo.
  Toto by som overila ako prvé: stačí `reset_index(drop=True)` po každom filtri.
- **[L]** `DELETE FROM links WHERE link_id IN (?...)` bez chunkovania na dvoch
  miestach ([normalization.py:691](src/sim/network/normalization.py#L691),
  [normalization.py:702](src/sim/network/normalization.py#L702)) — rovnaký limit
  premenných ako v kroku 1.
- **[D]** `CLAUDE.md` tvrdí, že `practical_speed.min_speed_kmh` je legacy kľúč a
  „ignoruje sa, ak je prítomný". V kóde sa číta
  ([normalization.py:310](src/sim/network/normalization.py#L310)) a používa ako
  spodná hranica clipu ([normalization.py:358](src/sim/network/normalization.py#L358)),
  ešte pred globálnym `thresholds.min_speed_kmh`. Buď opraviť dokumentáciu,
  alebo kľúč naozaj ignorovať — takto je to mína.
- **[L]** `_parse_spatialite_point` ([connectivity.py:441](src/sim/network/connectivity.py#L441))
  číta súradnice z BLOBu pevnými offsetmi 43:51 a 51:59. Funguje pre konkrétny
  formát SpatiaLite POINT; pri zmene verzie/SRID ticho vráti nezmysly (a použije
  sa to na výpočet vzdialeností pri spájaní slepých koncov).
- **[M]** Magické konštanty bez odkazu na zdroj: `_MIN_SPEED_RATIO = 0.4`
  ([normalization.py:796](src/sim/network/normalization.py#L796)),
  `peak_hour_factor = 0.10`, penalizačné hodnoty spojovacích liniek
  (20 km/h / 200 voz/h, [connectivity.py:306-308](src/sim/network/connectivity.py#L306-L308)),
  `max_snap_distance_m = 600`. Do práce patrí odôvodnenie alebo citácia.

**Čo zlepšiť**

1. Opraviť `geometry.length` a index alignment v CSD hintoch — obe sú lacné a
   obe sú typu „raz za čas ticho pokazí výsledok".
2. Rozhodnúť sa o CSD capacity hints: buď ich vypnúť a nechať kapacitu čisto
   tabuľkovú (a nechať ODME nech si poradí), alebo ich nechať a **explicitne to
   priznať vo validácii** — inak je časť dobrej zhody na týchto cestách
   tautológia.
3. Vytiahnuť magické čísla do `defaults.py` s komentárom odkiaľ sú (HCM, TP 189,
   ...). Zjednoduší to aj obhajobu.
4. Do `network_counts.json` pridať podiel `estimated_*` po triedach — dnes to
   vidno až v `supply_audit`, teda o 8 krokov neskôr.

---

## 4. `build-zones` — TAZ, brány, konektory

**Vstupy:** AequilibraE projekt (sieť), `zoning.sources` (OSM admin_level 9 /
katastre), `zoning.external_gateways`, `zone_population.parquet`.

**Algoritmus** ([zoning/pipeline.py:111](src/sim/zoning/pipeline.py#L111))

1. **AOI** — `build_model_area` (kvantil 1 %, padding 0.8 %, margin 200 m).
2. **Načítanie zón** zo zdrojov, filter „reprezentatívny bod v AOI",
   odstránenie prekryvov podľa `source_rank`.
3. **Brány (gateways)** ([zoning/gateways.py](src/sim/zoning/gateways.py), 1107 riadkov) —
   whitelist ciest z configu + `auto_discover_boundary_roads` (cesty pretínajúce
   hranicu AOI, filtrované podľa typu, počtu pruhov a blacklistu). Pre každú
   bránu sa vyberú cieľové uzly (`connectors_per_gateway`, default 2, minimálna
   vzájomná vzdialenosť 800 m) a vytvorí sa **syntetická zóna** mimo AOI
   (ID od 8 000 000 000).
4. **Centroidy** — `representative_point()` pre interné zóny (nie ťažisko —
   správne, ťažisko môže padnúť mimo zóny), uložené súradnice pre brány.
5. **Konektory** ([connectors.py:239](src/sim/zoning/connectors.py#L239)):
   - kandidátske uzly = uzly na „jazditeľnej" sieti, ktoré ležia v najväčšej
     **orientovanej** SCC
   - pre interné zóny sa vylučujú uzly, ktorých *všetky* incidentné linky sú
     `motorway`/`motorway_link` (aby zóna nevisela priamo na diaľnici)
   - výber je **sektorový** ([connectors.py:128](src/sim/zoning/connectors.py#L128)):
     okolie centroidu sa rozdelí na `min(max_connectors, 8)` uhlových sektorov,
     z každého sa vezme uzol s najvyšším skóre `road_weight / dist^0.75`,
     s vynúteným minimom jedného „major" konektora (v prípade potreby sa hľadá
     v 2.5× väčšom polomere)
   - konektor dostane 20 km/h, 2000 voz/h a **prístupovú penalizáciu 300 s**
     (brány len 5 s pri 90 km/h)
6. **Validácia konektorov** — varovania „príliš ďaleko" (> 1500 m) a
   „mimo SCC"; `connector_strict` to vie zmeniť na chybu.

**Výstupy:** `zones.geojson`, `centroids.geojson`, `zone_centroid_mapping.json`,
`gateway_diagnostics.csv`, `gateway_lookup_seed.parquet`, mapy.

**Rizikové miesta**

- **Zóny sú administratívne, nie dopravné.** Mestská časť Brno-střed má
  desaťtisíce obyvateľov a jeden centroid. Všetka doprava zóny „vzniká"
  v jednom bode a vteká do siete cez ≤ 6 konektorov — to je hlavný zdroj
  lokálnych chýb na uliciach v okolí centroidu (a dôvod, prečo model nemá
  zmysel čítať na úrovni jednej ulice).
- **Prístupová penalizácia 300 s je veľká páka.** Je to ~8 % z priemernej cesty
  a aplikuje sa na oba konce. Ovplyvňuje pomer krátkych a dlhých ciest
  v gravitačnom modeli aj to, ako veľmi sa oplatí obchádzka.
- **Brány sú bodové.** Celá diaľnica D1 zo západu vstupuje do modelu v jednom
  (resp. dvoch) uzloch. Kalibrácia brán potom pracuje s jedným číslom na koridor.

**Nálezy v kóde**

- **[L]** Prideľovanie ID centroidom hľadá prvé voľné celé číslo od 1
  ([connectors.py:323-330](src/sim/zoning/connectors.py#L323-L330)). Funguje, ale
  robí centroid ID nestabilné: pridanie jednej zóny prečísluje všetko a
  `zone_centroid_mapping.json` prestane sedieť na staršie matice
  (`.aem` sa indexuje centroid ID). Práve to je dôvod, prečo `warm_skims_regeneration_reason`
  musí kontrolovať rozmery matíc ([assignment/pipeline.py:56](src/sim/assignment/pipeline.py#L56)).
- **[L]** `_select_diverse_connectors` mieša indexy z dvoch rôznych DataFrame-ov
  (`nearby` a `cand`) a potom robí `all_pool.loc[valid_idx]`
  ([connectors.py:216-222](src/sim/zoning/connectors.py#L216-L222)) —
  funguje, lebo indexy sú zdedené z `eligible`, ale je to krehké a ťažko čitateľné.
- **[M]** `create_centroid_connectors` volá `eligible_road_nodes` **trikrát**
  (raz pre interné, dvakrát pre brány, [connectors.py:266](src/sim/zoning/connectors.py#L266)
  a [connectors.py:285-286](src/sim/zoning/connectors.py#L285-L286)), pričom prvé dve
  volania majú identické parametre. Čistá réžia (každé volanie stavia graf siete).

**Čo zlepšiť**

1. Odvodiť centroid ID deterministicky zo `zone_id` (napr. `zone_id + offset`)
   — odpadne celá trieda problémov so zosúladením matíc.
2. Zvážiť rozdelenie najväčších zón (Brno-střed, Brno-sever) na 2–3 podzóny.
   Je to najlacnejšie zlepšenie priestorového rozlíšenia, aké tento model môže
   dostať, a nevyžaduje žiadne nové dáta (populácia sa delí podľa plochy, tá
   logika už existuje).
3. Cachovať `eligible_road_nodes`.

---

## 5. `build-supernetwork` — tranzitná doprava

**Vstupy:** hrubá národná cestná sieť (OSM major roads), brány z kroku 4,
SLDB dochádzka **celá ČR**, centroidy obcí.

**Algoritmus** ([supernetwork/pipeline.py:57](src/sim/supernetwork/pipeline.py#L57))

1. Postaví sa **národný graf** hlavných ciest, voliteľne kontrahovaný
   (uzly stupňa 2 sa zlúčia, chránené uzly = brány a obce).
2. Brány a obce sa **nasnapujú** na najbližší uzol grafu.
3. `build_gateway_costs` — Dijkstra z/do každej brány → cena z brány do každej
   obce a späť; plus matica cien medzi bránami *cez model*.
4. `classify_relations` ([classification.py:21](src/sim/supernetwork/classification.py#L21)) —
   každá dvojica obcí z dochádzky sa klasifikuje:
   - obe interné → `internal_internal` (rieši `build-demand` priamo)
   - jedna interná → `external_internal` / `internal_external` + priradená brána
   - **obe externé** → kandidát na tranzit; akceptuje sa len ak
     ```
     cez_model / priamo ≤ detour_ratio_max   a   (cez_model − priamo) ≤ max_extra_minutes
     ```
     plus dve tvrdé pravidlá: rovnaká brána vstup/výstup sa zamieta a
     `reject_same_boundary_sector` zamieta dvojice brán v tom istom
     45° kompasovom oktante ([classification.py:16](src/sim/supernetwork/classification.py#L16)).
5. Akceptované relácie sa agregujú na dvojice brán → `through_gateway_pairs.parquet`.

**Rizikové miesta**

- **[M] Tranzit sa odvodzuje výhradne z dochádzky za prácou a do škôl.**
  Nákladná doprava, služobné jazdy, turistika a víkendová doprava v tom nie sú
  vôbec. Preto je nutný fudge faktor `through_traffic_scale` (Brno: 0.30,
  default 0.50) — ten v skutočnosti nekoriguje „koľko z dochádzkového tranzitu
  je reálne", ale kompenzuje **štrukturálne chýbajúci segment**. Toto by som
  v práci pomenovala explicitne; je to najslabšie miesto celého vstupu.
- **Jedna brána na obec.** `best_in` / `best_out`
  ([classification.py:56-65](src/sim/supernetwork/classification.py#L56-L65)) berie
  len bránu s rank 1. Reálne sa doprava z jednej obce rozdelí medzi viac
  vstupov (podľa cieľa v meste) — model to nezachytí.
- **Snapping na národný graf** — brána snapnutá > 500 m od grafu sa označí ako
  `large_snap_distance`, ale nezastaví beh.

**Čo zlepšiť**

1. Doplniť nákladnú dopravu z CSD: CSD má stĺpce `tv`/`t`/`pn`/`tn`, takže
   podiel ťažkých vozidiel na bránových profiloch je známy. Dá sa z toho urobiť
   samostatný tranzitný segment s vlastnou `pce` — a `through_traffic_scale` by
   prestal byť čiernou skrinkou.
2. Rozdeliť tranzit na 2–3 najlepšie brány s váhami (softmax podľa ceny),
   nie len rank 1.

---

## 6. `build-demand` — zostavenie OD matice

Podrobne rozobrané v [popis_simulacie.md](popis_simulacie.md#2-build-demand--konverzia-osôb-na-autá-tu-vzniká-najviac-voľnosti);
tu len to, čo je nad rámec parametrov.

**Algoritmus** ([demand/pipeline.py:38](src/sim/demand/pipeline.py#L38))

Štyri segmenty (`commuting`, `other`, `external_local`, `external_through`),
každý sa počíta zvlášť, rozdelí sa do 4 denných období pevnými podielmi a uloží
sa ako samostatný „core" v `.aem` matici + súčtový `wd_daily`.

Párovanie názvov obcí na zóny má tri režimy (`direct`, `group`, `lookup`) a
diagnostiku v `od_summary.json` (`mapped_direct`, `mapped_group`,
`mapped_external_lookup`, `missing_origin`, `missing_destination`).

**Rizikové miesta**

- **`skipped_external_external_rows: 237 400 z 253 377.** 94 % riadkov SLDB sa
  v `build-demand` zahodí (sú to dvojice mimo modelu) — správne, ale znamená to,
  že reálne pracujeme s ~15 600 dvojicami, z ktorých veľká časť má hodnotu
  jednotiek osôb. Štatistický šum SLDB (zaokrúhľovanie malých buniek kvôli
  ochrane údajov) sa priamo prenáša do matice.
- **Denné podiely období sú konštanty**, nie sú učené z CSD, hoci
  `learn-profile` (krok 13) presne takéto podiely počíta. Tie dva mechanizmy
  spolu nekomunikujú.

**Nálezy v kóde**

- **[M]** `_estimate_total_daily_trips_from_csd`
  ([seeds.py:132](src/sim/demand/seeds.py#L132)) delí súčet AADT „pokrytím"
  (`total / coverage`), čím extrapoluje na nespárované cesty. Pri nízkom pokrytí
  (napr. 2 zo 6 ciest) to znamená vynásobenie tromi — veľmi hrubý odhad
  s nulovým varovaním. V Brne sa nepoužíva (je tam pevné číslo), ale pre nové
  mestá je `auto` default.
- **[L]** `external_local` sa rozdeľuje medzi brány váhami podľa typu cesty a
  medzi interné zóny **podľa populácie** ([seeds.py:93-99](src/sim/demand/seeds.py#L93-L99)).
  Cieľová zóna vonkajšej dopravy ale koreluje skôr so zamestnanosťou / obchodmi
  než s počtom obyvateľov. Dáta o zamestnanosti sú pritom k dispozícii.

**Čo zlepšiť**

1. Prepojiť `learn-profile` → `demand.time_slices` (dnes sa profil počíta a
   nikam sa nevracia).
2. `external_local` rozdeľovať váženým priemerom populácie a zamestnanosti.

---

## 7. `assign-warm-skims` — predbežné priradenie

**Vstupy:** projekt + OD matica.

**Algoritmus** ([assignment/pipeline.py:285](src/sim/assignment/pipeline.py#L285)) —
to isté ako `assign`, ale `max_iter=30`, `rgap_target=0.01`, `save_skims=True`,
`skim_method="final"` (skim z posledného all-or-nothing priechodu, nie
priemerovaný). Nekonvergovanie sa tu vedome prehltne a zaloguje ako info.

**Riziko:** skimy z rgap 0.01 sú „časy v sieti, ktorá ešte nie je v rovnováhe".
`distribute` na nich fituje β gravitačného modelu. Kód to stráži
(`_validate_skim_convergence`, [impedance.py:33](src/sim/distribution/impedance.py#L33))
— ale kontroluje len flag `converged`, ktorý je pri rgap_target 0.01
splnený triviálne. Prah 0.01 sa zároveň používa ako „varovná" hranica
([impedance.py:18](src/sim/distribution/impedance.py#L18)), takže varovanie
sa nikdy nespustí (`rgap > 0.01` je pri cieli 0.01 nepravda).

**Čo zlepšiť:** znížiť `warm_skim_pass.rgap_target` na ~0.005 (beh je krátky) a
oddeliť „cieľ" od „varovného prahu".

---

## 8. `distribute` — gravitácia + IPF

**Vstupy:** OD matica, `skims.aem`, `zone_population.parquet`,
`zone_employment.parquet`.

**Algoritmus** ([distribution/pipeline.py](src/sim/distribution/pipeline.py))

Pre každý segment v `demand.distribution.segments` (default len `other`):

1. `calibrate_gravity_simple` ([gravity.py:9](src/sim/distribution/gravity.py#L9)) —
   AequilibraE `GravityCalibration` (EXPO) na seed matici + skim matici.
2. Ak `beta < min_gravity_beta` (0.0001) → segment sa **preskočí**
   (impedancia nemá vplyv, seed zostáva).
3. `build_pa_vectors` ([pa_vectors.py:10](src/sim/distribution/pa_vectors.py#L10)):
   ```
   production[i] = populácia[i] × pa_trip_rate × pa_car_share / pa_occupancy
   attraction[j] = zamestnanosť[j] × (Σ production / Σ zamestnanosť)
   ```
   Atrakcie sa teda **preškálujú na súčet produkcií** — IPF má konzistentné
   okraje, čo je správne.
4. `run_ipf` ([gravity.py:62](src/sim/distribution/gravity.py#L62)) na matici
   `exp(−β × impedancia)` s cieľovými okrajmi P/A.
5. **Blend** `alpha × IPF + (1−alpha) × seed`, `alpha = 0.55`.
6. **Strop** `max_total_multiplier = 3.5` × seed.
7. `wd_daily` sa **prepočíta ako súčet segmentov**.

**Nálezy v kóde**

- **[M] — najdôležitejší nález celého rozboru.** Segment `other` je
  vygenerovaný v kroku 6 z jednej sady parametrov
  (`trip_rate=1.0`, `car_share=0.35`, `occupancy=1.5` → **0.233 cesty na obyvateľa**)
  a v kroku 8 sa mu vnútia okraje z **úplne inej** sady
  (`pa_trip_rate=1.8`, `pa_car_share=0.40`, `pa_occupancy=1.3` → **0.554 cesty na obyvateľa**),
  teda 2.37×. Po blende 55/45 z toho vyjde ~1.75× a segment narastie z
  93 035 na ~164 000 vozidiel/deň. Overené na reálnom behu: celkový dopyt
  461 374 → 532 412, rozdiel 71 038 ≈ presne tento nárast.

  Inými slovami: **parametre segmentu `other` z `build-demand` sú v podstate
  irelevantné** — určujú len tvar matice, mieru prepíše `distribute`. To je
  v poriadku ako návrh (P/A vektory sú „pravda", seed len tvar), ale musí to
  byť vedomé rozhodnutie a musí to byť v práci napísané. Dnes to vyzerá ako
  dva nezávislé odhady toho istého čísla, ktoré si nikto nevšimol, že si
  odporujú.

- **[B]** `calibrate_gravity_simple` aj `run_ipf` majú okolo celého
  AequilibraE volania `try: ... except Exception: pass`
  ([gravity.py:46](src/sim/distribution/gravity.py#L46),
  [gravity.py:93](src/sim/distribution/gravity.py#L93)) a ticho spadnú do
  vlastného fallbacku. Nikde sa nezaloguje, ktorá vetva bežala. Fallbacky sú
  pritom výrazne hrubšie:
  - gravitačný fallback fituje `log(T) ~ c` obyčajnou `polyfit` regresiou
    ([gravity.py:49-59](src/sim/distribution/gravity.py#L49-L59)), teda **bez**
    členov produkcie a atrakcie — odhad β je vychýlený
  - IPF fallback (Furness) používa **absolútnu** toleranciu 0.001 vozidla
    ([gravity.py:108-111](src/sim/distribution/gravity.py#L108-L111)); pri
    súčtoch v státisícoch sa nikdy nesplní, takže beží vždy všetkých 200
    iterácií (nevadí numericky, vadí to, že „konvergovalo" nič neznamená)

  Minimum: `except Exception: logger.warning(..., exc_info=True)` a zápis
  použitej vetvy do `distribution_report.json`.

- **[M]** Strop `max_total_multiplier` je zdôvodnený komentárom „keď chýbajú
  dáta o zamestnanosti" ([pipeline.py:243-248](src/sim/distribution/pipeline.py#L243-L248)),
  ale aplikuje sa vždy. Ak by nastal, celý segment sa preškáluje konštantou —
  a v reporte je len `capped: true`. Malo by to byť hlasnejšie.

**Čo zlepšiť**

1. Zjednotiť parametre: buď `other` generovať rovno s `pa_*` hodnotami (a seed
   používať čisto ako tvar), alebo `pa_*` odvodiť zo segmentových parametrov.
   Dnešný stav je nechtiac dvojitý model.
2. Odstrániť tiché fallbacky.
3. Do reportu pridať porovnanie `seed_total` / `ipf_total` / `blended_total`
   **po zónach**, nie len globálne — tam by bolo hneď vidno, ktoré zóny IPF
   nafúkol.

---

## 9. `assign` — priradenie na sieť

**Vstupy:** projekt (sieť + konektory), OD matica po distribúcii.

**Algoritmus** ([assignment/executor.py:55](src/sim/assignment/executor.py#L55),
graf v [assignment/graph.py:12](src/sim/assignment/graph.py#L12))

1. **Graf:** `build_graphs(modes=["c"])` + `prepare_graph(centroid_ids)`;
   cena hrany = `free_flow_time`; `set_blocked_centroid_flows(True)` (trasa
   nesmie prejsť cez cudzí centroid).
2. **Denná kapacita:** hodinová kapacita sa násobí faktorom podľa typu cesty
   (diaľnica 13, obytná ulica 8, [graph.py:71-97](src/sim/assignment/graph.py#L71-L97)),
   aby BPR videlo zmysluplné V/C pri dennej matici. Inverzia faktora sa uloží
   ako `_k_factor` na spätný prepočet špičkovej hodiny.
3. **Triedy:** dve `TrafficClass` nad tým istým grafom — `local` (vot 1.0,
   pce 1.0) a `through` (vot 2.0, pce 1.5). Matica triedy sa škáluje pomerom
   `wd_daily / Σ(triedy)` po bunkách
   ([executor.py:129-141](src/sim/assignment/executor.py#L129-L141)), aby súčet
   tried sedel na denný core.
4. **UE:** BFW (biconjugate Frank–Wolfe), `max_iter=200`, `rgap_target=0.002`.
   All-or-nothing je **zakázané** a hodí `ValueError`
   ([executor.py:26](src/sim/assignment/executor.py#L26)) — dobré rozhodnutie,
   dobre zdokumentované.
5. **Hľadanie trás:** Dijkstra one-to-all z každého centroidu (Cython
   implementácia v AequilibraE), paralelne cez originy
   ([executor.py:217](src/sim/assignment/executor.py#L217)).
6. **Select-link** (voliteľne) — sleduje, ktoré OD dvojice použili daný link;
   používa to kalibrácia brán.

**Rizikové miesta**

- **Denná matica na hodinových kapacitách.** Faktor 10–13 je štandardný trik,
  ale znamená, že model nemá špičku — má „priemerný deň rozotretý". Kongescia
  vychádza z toho, že *denný* objem prekročí *dennú* kapacitu, čo nie je to
  isté ako reálne zápchy v 8:00. Pre denné objemy (čo je výstup) to je
  akceptovateľné; pre tvrdenia o zdržaní už menej.
- **Žiadne odbočenia ani križovatkové zdržania.** Sú „rozmazané" do rýchlosti
  cez `ipkm` v kroku 3. Model teda nevie, že ľavé odbočenie cez hlavnú trvá
  dlhšie než priame.
- **Žiadna stochastická voľba trasy.** Deterministické Wardropovo UE
  predpokladá dokonalú informáciu a homogénne správanie. Reálne sa doprava
  rozteká viac než UE predpovedá (preto býva model na paralelných koridoroch
  „príliš ostrý"). Repozitár na to má diagnostiku
  ([diagnostics/parallel_corridor.py](src/sim/diagnostics/parallel_corridor.py)) —
  dobre.
- **Bit-level nedeterminizmus** z viacvláknového sčítania float čísel
  (jediné miesto v celej pipeline).

**Nálezy v kóde**

- **[L]** `skim_method="final"` siaha na privátny atribút
  `primary_tc._aon_results` ([executor.py:277](src/sim/assignment/executor.py#L277)).
  Je tam fallback na verejné `results.skims`, takže to nespadne, ale pri zmene
  AequilibraE API sa ticho zmení význam skimov (posledný AoN vs. priemerovaný
  skim), čo posunie β v `distribute`.
- **[M]** Multi-class škálovanie predpokladá, že `wd_daily` = súčet
  `wd_daily_local` + `wd_daily_external_through`. Ak sa `distribute` dotkne len
  `other` a `wd_daily` sa prepočíta, sedí to; ale je to nepísaný invariant
  medzi dvomi modulmi. Stálo by za to ho testom uzamknúť.

**Čo zlepšiť**

1. Zvážiť SUE (stochastic UE) alebo aspoň C-logit ako citlivostný experiment —
   ukázalo by, koľko z chyby na paralelných koridoroch je metodika a koľko dáta.
2. Zapísať `_k_factor` a použité `daily_capacity_factor` do
   `assignment_results.parquet`, aby bol prepočet na špičku dohľadateľný
   bez configu.

---

## 10. `audit-supply` — kontrola ponukovej strany

**Algoritmus** ([supply_audit.py:57](src/sim/calibration/supply_audit.py#L57)) —
agregácia rýchlostí/kapacít/pruhov po `link_type` a porovnanie s tabuľkami
očakávaných rozsahov ([supply_audit.py:22](src/sim/calibration/supply_audit.py#L22),
[supply_audit.py:39](src/sim/calibration/supply_audit.py#L39)), plus podiel
imputovaných hodnôt (varovanie nad 70 %) a V/C diagnostika, ak už existujú
výsledky priradenia.

**Rizikové miesta**

- **Verdikt je vždy PASS.** Buď `PASS`, alebo `PASS_WITH_WARNINGS`
  ([supply_audit.py:190](src/sim/calibration/supply_audit.py#L190)) — nikdy `FAIL`.
  Krok je pritom povinná brána pred ODME, takže brána nikdy nič nezastaví.
  Jediné, čo robí, je že vytvorí súbor, ktorého existenciu kalibrácia kontroluje.
- Audit sa pozerá len na **priemer** za triedu (`AVG(speed_ab)`). Trieda s
  polovicou liniek na 5 km/h a polovicou na 95 km/h prejde ako OK.
- Kontrola je len na `*_ab` stĺpce — asymetria smerov sa neaudituje.

**Čo zlepšiť**

1. Pridať skutočný `FAIL` (napr. > 90 % imputovaných rýchlostí, alebo medián
   mimo rozsahu) a nechať `require_supply_audit` rozhodnúť, či to zastaví beh.
2. Auditovať percentily (p5/p50/p95), nie priemery, a oba smery.

---

## 11. `calibrate` — ODME

**Vstupy:** OD matica po distribúcii, `assignment_results.parquet`, CSD sčítania
(kalibračná časť), screenline definície, `supply_audit.json`.

**Algoritmus** ([odme.py:117](src/sim/calibration/odme.py#L117),
kontext v [context.py](src/sim/calibration/context.py))

Bi-level slučka, max. 48 vonkajších iterácií, rozdelená na stage
(`stabilize` 20, `refine` 35), s per-stage prepisom parametrov:

**Dolná úroveň (raz za iteráciu):** UE priradenie s `assign_max_iter=120`,
`assign_rgap_target=0.004` + select-link matice pre každý screenline.

**Horná úroveň (poradie záleží):**

1. **Screenline aktualizácia** — pre každý screenline `ratio = obs / mod`;
   ak je pomer mimo `[sl_ratio_min, sl_ratio_max]`, screenline sa preskočí.
   Inak sa každá bunka matice vynásobí
   ```
   1 + damping × (ratio − 1) × proportion[i,j]
   ```
   kde `proportion` = podiel danej OD bunky, ktorý cez screenline reálne ide
   (zo select-link matice). `damping = 1/√(počet aktívnych screenlinov)`.
   Opakuje sa `gradient_descent_iterations` (8) krát.
   ([odme.py:27](src/sim/calibration/odme.py#L27))
2. **Globálny reziduál** ([context.py:791](src/sim/calibration/context.py#L791)) —
   celá matica × `1 + damping × (Σobs/Σmod − 1)`, orezané na `[min, max]`.
3. **Reziduál po triedach ciest** — to isté, ale po `link_type`.
4. **Kalibrácia brán** ([context.py:853](src/sim/calibration/context.py#L853)) —
   objem na koridore sa dorovná na pozorované AADT faktorom v `[0.60, 1.45]`
   s dampingom 0.20. Po prvej iterácii sa (voliteľne) **prepíšu hranice
   elasticity** na nový stav.
5. **Elasticita** — po každom kroku `clip(demand, seed/4, seed×4)`.
6. **Strop na iteráciu** — zmena celkového dopytu max. ±12 %.

**Sledovanie kvality:** objektívna funkcia `Z` (vážený súčet štvorcov odchýlok
s váhami `1/√obs`), tracking najlepšej iterácie, návrat na najlepší stav po
3 zhoršeniach, ukončenie pri `|ΔZ|/Z < 0.0008`, stagnácii (20 iterácií) alebo
splnení všetkých denných kritérií.

**Rizikové miesta — toto je najcitlivejšia časť práce**

- **[M] Vylučovanie zle sediacich sčítaní.** Pred kalibráciou beží
  `match_counts_to_links`, ktorá zo vzorky vyradí:
  - stanice s pomerom `mod/obs` mimo `[0.20, 5.0]`
    ([matching.py:38](src/sim/calibration/matching.py#L38))
  - stanice s nízkou „confidence" — pri geometricky neistom priradení sa do
    skóre počíta aj pomer objemov ([matching.py:285](src/sim/calibration/matching.py#L285))
  - **ručný zoznam 12 čísel ciest** v configu
    ([config/brno/sim.yaml:80](config/brno/sim.yaml#L80))

  Vylúčené stanice sa v prvej iterácii **natrvalo odstránia** zo vzorky
  ([context.py:656-673](src/sim/calibration/context.py#L656-L673)) a
  `filter_matched_counts_for_benchmark` ich vyhodí aj z R², RMSE, bias a GEH.
  Časť vylúčení je legitímna (zle spárovaná geometria, linka odpojená od grafu).
  Ale kritérium „pomer mimo [0.2, 5]" **nerozlišuje zlé spárovanie od zlého
  modelu** — a keďže sa aplikuje na prvom priradení, vylučuje presne tie
  pozorovania, ktoré model netrafil. Reportované metriky sú preto optimistické
  a nie sú porovnateľné s prácami, ktoré filtrujú len geometricky.

  **Odporúčanie do práce:** uviesť obe čísla — metriky na plnej vzorke aj na
  filtrovanej — a zoznam vylúčení s dôvodom pre každý úsek. Je to pár riadkov
  navyše a úplne to odzbrojí námietku.

- **[M] Globálny reziduál je silný predpoklad.** Škálovať celú maticu (461 tis.
  ciest) faktorom nameraným na ~14 úsekoch znamená predpokladať, že tie úseky
  reprezentujú celkový výkon dopravy v meste. CSD meria hlavne cesty I. a II.
  triedy — mestské zbernice a obslužné komunikácie v ňom skoro nie sú. Model sa
  teda „doťahuje" na hlavnú sieť a rovnaký faktor dostane aj doprava po
  uliciach, ktoré nikto nemeral.

- **[M] Poradie screenlinov ovplyvňuje výsledok.** `_spiess_update_step`
  aplikuje screenline sekvenčne a každý ďalší už vidí maticu zmenenú
  predchádzajúcim ([odme.py:47-67](src/sim/calibration/odme.py#L47-L67)).
  Poradie je poradie vloženia do dictu (deterministické, ale arbitrárne).
  Korektnejšie by bolo zozbierať všetky korekcie a aplikovať ich naraz.

- **[D] „Spiess" nie je Spiess.** Spiessova metóda (1990) počíta gradient
  objektívnej funkcie podľa OD buniek cez maticu proporcií priradenia a hľadá
  optimálnu dĺžku kroku line-searchom. Tu je multiplikatívna heuristika
  s pevným dampingom `1/√n` a bez line-searchu. Je to príbuzné (proporcie sa
  používajú rovnako), ale nazvať to „Spiess gradient method" je v práci
  napadnuteľné. Navrhujem prepísať na „proporcionálnu ODME aktualizáciu
  inšpirovanú Spiessom" — alebo doplniť skutočný line-search, čo je pri
  existujúcej infraštruktúre ~30 riadkov.

- **[M] Kalibrácia brán mení výsledok viac než ODME.** Faktory do 1.45 na
  koridor s dampingom 0.20, aplikované na 5+ brán, ovplyvnia celkový dopyt
  o desiatky percent — a to ešte pred prvým gradientovým krokom.
  Poradie „brány → rebase elasticity" znamená, že hranice `seed/4 … seed×4`
  sa po prvej iterácii počítajú už z **upravenej** matice
  ([context.py:874-882](src/sim/calibration/context.py#L874-L882)), čiže reálna
  povolená odchýlka od pôvodného seedu je väčšia než deklarovaných 4×.

**Nálezy v kóde**

- **[L]** Flag `converged` v `calibration_report.json` je OR troch podmienok
  ([odme.py:555-570](src/sim/calibration/odme.py#L555-L570)), vrátane „posledné
  dve iterácie sa líšia o menej než tolerancia". Stagnácia na zlom riešení sa
  teda vykáže ako konvergencia. V reporte by mal byť dôvod ukončenia
  (`stop_reason` už v FSM existuje — [context.py:760](src/sim/calibration/context.py#L760) —
  len sa nezapisuje).
- **[L]** `save_skims` sa v ODME robí len v prvej iterácii
  ([odme.py:247](src/sim/calibration/odme.py#L247)); ak by ich niekto použil pre
  `distribute`, dostal by časy z nekalibrovaného modelu.

**Čo zlepšiť**

1. Rozdeliť vylúčenia na **geometrické** (legitímne, vylúčiť) a **objemové**
   (podozrivé, ponechať a reportovať zvlášť). Toto je jediná zmena, ktorá
   podstatne zvýši dôveryhodnosť výsledkov.
2. Zapisovať `stop_reason` a zoznam vylúčených úsekov s dôvodom do reportu.
3. Aplikovať screenline korekcie naraz (zbierať multiplikátory, potom násobiť).
4. Premenovať metódu alebo doplniť line-search.

---

## 12. `validate` — nezávislé overenie

**Vstupy:** výsledky priradenia, CSD (validačná časť), screenline.

**Algoritmus** ([validation.py:1029](src/sim/calibration/validation.py#L1029))

CSD sa rozdelí na kalibračnú a validačnú časť
(`split_csd_for_calibration`, [observed.py:441](src/sim/calibration/observed.py#L441)),
stratégia `corridor`. Metriky (R², slope, %RMSE, bias, GEH, Spearman) sa počítajú
zvlášť pre obe časti a **verdikt PASS/FAIL sa berie z holdoutu**
([validation.py:826](src/sim/calibration/validation.py#L826)) — to je správne
a je to silná stránka práce.

**Rizikové miesta**

- **[M] „Corridor" split nie je náhodný.** Napriek `random_seed` sa cesty
  v každej triede zoradia **podľa počtu úsekov zostupne** a greedy sa napĺňa
  kalibračná časť, kým nedosiahne 65 % úsekov
  ([observed.py:554-568](src/sim/calibration/observed.py#L554-L568)). Zamiešanie
  RNG rozhodne len o poradí ciest s rovnakým počtom úsekov. Dôsledok:
  **do kalibrácie idú systematicky najdlhšie/najvýznamnejšie koridory a do
  holdoutu zvyšky**. Holdout preto nie je reprezentatívny — je posunutý smerom
  ku kratším a menej zaťaženým cestám. Nezávislosť (žiadna cesta v oboch
  množinách) je zachovaná a to je hlavné, ale tvrdenie „náhodný split" by
  v práci nemalo zaznieť.
- **[M] Holdout nie je celkom nedotknutý.** Config má
  `csd_validation_exclude_sil` (9 ciest) navyše k `exclude_csd_roads` (12 ciest)
  ([config/brno/sim.yaml:79-80](config/brno/sim.yaml#L79-L80)). Ručný zásah do
  validačnej vzorky je presne ten typ veci, na ktorý sa oponent pýta.
- **Veľkosť holdoutu.** `_classify_holdout_adequacy`
  ([validation.py:775](src/sim/calibration/validation.py#L775)) klasifikuje
  < 5 úsekov ako „insufficient", 5–9 ako „thin". Pri ~14 použiteľných úsekoch
  celkovo je holdout na hranici — jeden zle spárovaný úsek posunie R² o desatiny.
- **GEH** sa počíta, ale pre denný model nedáva zmysel (navrhnutý pre hodinové
  toky) — kód to vie a vylučuje ho z konvergenčného kritéria
  ([context.py:765](src/sim/calibration/context.py#L765)). Dobre.

**Čo zlepšiť**

1. Zmeniť `corridor` split tak, aby v rámci triedy náhodne vyberal cesty do
   naplnenia kvóty (namiesto zoradenia podľa veľkosti) — a spustiť validáciu
   pre 5–10 seedov, výsledok reportovať ako rozptyl. To je najsilnejší možný
   argument o robustnosti a stojí to jeden cyklus behov.
2. Odstrániť `csd_validation_exclude_sil` alebo pre každý úsek doložiť dôvod.

---

## 13.–15. `learn-profile`, `strip-closures`, `serve`

### `learn-profile`

**Algoritmus** ([temporal.py:65](src/sim/demand/temporal.py#L65)) — z CSD sa
počítajú denné faktory (pracovný deň / sobota / nedeľa / sviatok) a podiely
deň/večer/noc.

**Nálezy**

- **[B]** Denné faktory sa počítajú ako **priemer pomerov**
  `mean(ipd_o / o)` ([temporal.py:90-91](src/sim/demand/temporal.py#L90-L91)).
  Správne má byť **vážený** pomer `Σ ipd_o / Σ o` — inak úsek s 200 vozidlami
  denne váži rovnako ako D1 s 60 000 a priemer ťahajú malé, zašumené úseky.
  Oprava je jednoriadková.
- **[B]** Sviatky sú zoznam `(mesiac, deň)`
  ([temporal.py:24](src/sim/demand/temporal.py#L24)) — pohyblivé sviatky
  (Veľký piatok, Veľkonočný pondelok) sa nedajú vyjadriť a chýbajú.
- **[B]** `_holidays_md_cache` je **modulová globálna premenná**
  ([temporal.py:26](src/sim/demand/temporal.py#L26)) naplnená pri prvom volaní.
  Pri behu viacerých miest v jednom procese (`experiments/run_all.py`, API)
  druhé mesto ticho dostane sviatky prvého.
- **[M]** Sobota vs. nedeľa sa nerozlišuje z dát (CSD ich nemá zvlášť) — počíta
  sa z víkendového faktora násobičmi 1.10 / 0.90 z configu. Poctivé, ale
  je to predpoklad, nie meranie.
- Výstup `temporal_profile.json` sa nikam nevracia — `demand.time_slices` ostáva
  na konštantách z `defaults.py`.

### `strip-closures`

Vráti `_preclosure_*` hodnoty späť a zahodí stĺpce
([closures.py:60](src/sim/network/closures.py#L60)).

**[M] Metodické riziko:** matica bola kalibrovaná na sieti **s uzávierkami**.
ODME časť rozdielu medzi modelom a sčítaním nutne zapísalo do dopytu (presunulo
cesty inam, aby sedeli objemy na obchádzkových trasách). Po odstránení uzávierok
tá deformácia v matici zostáva. Pre scenáre to znamená, že „baseline" nie je
čistý stav siete bez uzávierok, ale stav siete bez uzávierok s dopytom
prispôsobeným uzávierkam. V Brne je `baseline_closures.enabled: false`, takže
dnes to nehrozí — ale ak sa to zapne, treba po strip-closures prebehnúť aspoň
`assign` znovu a povedať to v práci.

### `serve`

FastAPI, len na čítanie ([api.py](src/sim/api.py), ~28 endpointov). GeoDataFrame-y
sa lazy-loadujú ako singletony kľúčované `mtime` súboru.

**Scenáre** ([scenarios/engine.py:43](src/sim/scenarios/engine.py#L43)):
uzávierka nastaví linke `CLOSURE_CAPACITY` a obrovský čas jazdy (link sa
nemaže — štandardný postup), čiastočné zúženie násobí kapacitu pomerom
zostávajúcich pruhov. Beží v background threade nad **kalibrovanou** maticou.

**Riziká**

- **Scenáre menia len voľbu trasy.** Žiadna elasticita dopytu — nikto necestu
  nezruší, neprejde na MHD, neposunie odchod. Pri veľkých uzávierkach model
  preto systematicky nadhodnocuje objemy na obchádzkach. Toto treba pri každom
  scenárovom výsledku uviesť.
- **[L]** Zúženie mení len kapacitu, nie voľný čas jazdy. Fyzikálne to je
  obhájiteľné (spomalenie príde cez BPR), ale znamená to, že zúženie na
  prázdnej ceste nemá **žiadny** efekt.
- Joby žijú v pamäti procesu (`_jobs_lock`, `_prune_old_jobs`) — reštart API
  ich stratí.

---

## Prierezové veci

### Čo je na tomto modeli dobré (aby to v práci nezaniklo)

- **All-or-nothing je tvrdo zakázané** a s odôvodnením priamo v kóde.
  Veľa diplomových modelov to takto nemá.
- **Verdikt validácie sa berie z holdoutu**, nie z kalibračnej vzorky.
- **Elasticita ODME je obmedzená** (seed/4 … seed×4) a odchýlka od seedu sa
  reportuje vrátane 10 najviac zmenených buniek
  ([odme.py:480-544](src/sim/calibration/odme.py#L480-L544)).
- **Idempotentná normalizácia** cez `posted_speed_*` — opakovaný beh nedegraduje
  rýchlosti (klasická chyba, tu vyriešená a dokonca s detekciou starých behov).
- **Supply audit ako brána pred ODME** s explicitným zdôvodnením, že kalibrácia
  dopytu nemá zakrývať chyby siete.
- Konzistentná telemetria: každý krok píše JSON s profilingom a štatistikami.

### Tri veci, ktoré by som opravila ako prvé

1. **Zjednotiť parametre segmentu `other`** (krok 8, nález č. 1) — dnes model
   obsahuje dva nezosúladené odhady toho istého a jeden ticho vyhráva.
2. **Rozdeliť vylučovanie sčítaní na geometrické a objemové** (krok 11,
   nález č. 2) a reportovať metriky na oboch vzorkách.
3. **Náhodný `corridor` split + viac seedov** (krok 12, nález č. 3) —
   z „holdout vyzerá dobre" sa stane „holdout vyzerá dobre pri ľubovoľnom
   rozdelení", čo je úplne iná sila tvrdenia.

### Čo by som naopak neriešila

- Latentné SQLite/index nálezy (`[L]`) — opraviť pri najbližšom dotyku súboru,
  nie kvôli nim rozbíjať funkčný beh.
- Výkon (`iterrows` na desiatkach tisíc liniek na viacerých miestach) — beh
  trvá minúty, nie hodiny, a nie je to predmetom práce.
