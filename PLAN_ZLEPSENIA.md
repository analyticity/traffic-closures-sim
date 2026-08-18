# Plán zlepšenia modelu

> Nadväzuje na [pipeline_detail.md](pipeline_detail.md) (technický rozbor krokov)
> a [KAMDOJIZDIME_PLAN.md](KAMDOJIZDIME_PLAN.md) (mobilné dáta).
> Tento dokument je **prioritizovaný akčný plán**: čo opraviť, v akom poradí,
> aké je to rizikové a ako sa to zmeria.

**Dátum:** 17. 8. 2026 · **Baseline pre porovnanie:** `outputs_baseline_v0/brno/baseline/`
(R² holdout 0.700, slope 0.715, %RMSE 42.2, bias −18.9 %, n=14)

---

## Obsah

- [Diagnóza: tri prepojené problémy](#diagnóza-tri-prepojené-problémy)
- [Prioritizovaný zoznam opráv](#prioritizovaný-zoznam-opráv)
- [P0 — najprv](#p0--najprv)
- [P1 — potom](#p1--potom)
- [P2 — keď bude čas](#p2--keď-bude-čas)
- [Rola kamdojizdime](#rola-kamdojizdime)
- [Navrhované poradie prác](#navrhované-poradie-prác)
- [Čo nerobiť](#čo-nerobiť)

---

## Diagnóza: tri prepojené problémy

Väčšina toho, čo v modeli nesedí, sa dá vysvetliť tromi vecami, ktoré sa navzájom
podporujú. Nie sú to nezávislé chyby — je to jeden reťazec.

### A. Model má 5 dverí, mesto ich má 20

Externá doprava môže do modelu vojsť len cez 5 bránových zón
(D1 západ, D1 východ, D2 juh, I/52 juh, I/43 sever). Z CSD 2025 (úseky
označené hranicou okresu Brno-město / Brno-venkov):

```
brány (5 screenlinov, observed):      230 213 voz/deň
hraničné cesty BEZ brány:             137 093 voz/deň   ← 37 %
```

Najväčšie chýbajúce: **I/50** (24 271, východ), **II/602** (18 523, západ),
**II/380** (12 348, juhovýchod), **6401** (10 998, sever),
**II/430** (10 115, východ), **15286** (9 977, Šlapanice).

Dôsledok na objemy — porovnanie modelového maxima na ceste s CSD AADT na hranici
(platí len pre nerozdelené cesty II./III. triedy; u rozdelených sa objem delí
medzi dva jazdné pásy, takže tam je pomer nepoužiteľný):

| cesta | CSD hranica | model max | pomer |
|---|---:|---:|---:|
| II/430 | 10 115 | 4 565 | **0.45** |
| 15286 | 9 977 | 5 012 | **0.50** |
| 15276 | 4 854 | 2 643 | **0.54** |
| 6401 | 10 998 | 6 081 | **0.55** |
| 41614 | 3 357 | 2 159 | **0.64** |
| II/380 | 12 348 | 9 267 | **0.75** |
| II/602 (Jihlavská) | 18 523 | 14 946 | **0.81** |
| 3846 | 7 430 | 6 307 | 0.85 |

Systematicky pod — presne ako predpovedá hypotéza chýbajúcich brán.

### B. Kalibrácia ten problém maskuje, nerieši

V configu je ručne vylúčených 12 ciest z kalibrácie a 9 z validácie.
**7 z tých 12** (602, 430, 380, 373, 3846, 41614, 15278) sú presne hraničné cesty
bez brány. Model na nich nemôže mať pravdu — nemajú kadiaľ dostať externú
dopravu. Ich vylúčenie nelieči zlé geometrické párovanie, ale odstraňuje
dôkaz o probléme A.

K tomu sa pridáva automatické vylúčenie staníc s pomerom `mod/obs` mimo
[0.2, 5] ([matching.py:38](src/sim/calibration/matching.py#L38)) — vylučuje sa
podľa toho, ako veľmi sa model mýli.

### C. Segment `other` je modelovaný dvakrát

`build-demand` ho vyrobí s jednou sadou parametrov (0.233 cesty/obyv.),
`distribute` mu vnúti okraje z druhej sady (0.554 cesty/obyv.) — 2.37×.
Po blende narastie z 93 tis. na ~164 tis. voz/deň (overené: celkový dopyt
461 374 → 532 412). Prvá sada je fakticky mŕtva.

### Ako to spolu súvisí

```
A: doprava vojde nesprávnou bránou
        ↓
   radiály bez brány sú podhodnotené, brány preťažené
        ↓
B: kalibrácia brány pripne na observed a prebytok zmaže z matice
   (532 412 → 384 869, bias −18.9 %)
        ↓
   podhodnotenie sa rozleje po celom meste
        ↓
C: nekonzistentná generácia dopytu to celé ešte rozmazáva
```

**Preto sa oplatí riešiť A ako prvé.** B a C sa dajú opraviť nezávisle,
ale kým platí A, nebude sa dať zmerať, či pomohli.

---

## Prioritizovaný zoznam opráv

Legenda:
**Riziko** = ako veľmi to dnes skresľuje výsledky ·
**Práca** = odhad ·
**Riziko zmeny** = šanca, že oprava rozbije niečo iné

| # | Oprava | Krok | Riziko | Práca | Riziko zmeny |
|---|---|---|---|---|---|
| **P0-1** | Pridať brány na I/50, II/602, II/380, II/430, 6401, 15286 | 4 | 🔴 vysoké | 1 h (YAML) + behy | stredné |
| **P0-2** | Zrušiť ručné `exclude_csd_roads`, rozdeliť vylúčenia na geometrické vs. objemové | 11 | 🔴 vysoké | 3 h | nízke |
| **P0-3** | Zjednotiť parametre segmentu `other` (`pa_*` vs. seed) | 6+8 | 🔴 vysoké | 2 h | stredné |
| **P0-4** | Overiť kotvenie brány I/43 (observed 22 478 vs. CSD hranica 40 582) | 4+11 | 🔴 vysoké | 2 h | nízke |
| **P0-5** | Opraviť opravnú linku 117130 (V/C 12.9, cestovný čas 76 h) | 3 | 🔴 vysoké | 1 h | nízke |
| **P1-1** | Náhodný `corridor` split + validácia cez viac seedov | 12 | 🟠 stredné | 2 h + behy | nízke |
| **P1-2** | Zamestnanosť vrátane `lokalizace=0` (pracujúci v mieste) | 2 | 🟠 stredné | 1 h | stredné |
| **P1-3** | Párovanie zón na obce cez kód obce, nie názov; zrušiť `default_median` | 2 | 🟠 stredné | 4 h | nízke |
| **P1-4** | Odstrániť tiché fallbacky v gravitácii/IPF | 8 | 🟠 stredné | 30 min | žiadne |
| **P1-5** | Vážené denné faktory v `learn-profile` (Σ/Σ namiesto mean pomerov) | 13 | 🟠 stredné | 15 min | nízke |
| **P1-6** | Nákladná doprava do tranzitu z CSD (`tv`) namiesto `through_traffic_scale` | 5 | 🟠 stredné | 1 deň | stredné |
| **P2-1** | `geometry.length` → metrický CRS | 3 | 🟡 latentné | 10 min | žiadne |
| **P2-2** | `reset_index` pred CSD capacity hints | 3 | 🟡 latentné | 10 min | žiadne |
| **P2-3** | Chunkovanie `DELETE ... IN (?)` | 1+3 | 🟡 latentné | 20 min | žiadne |
| **P2-4** | `_holidays_md_cache` — zrušiť globálnu cache | 13 | 🟡 latentné | 10 min | žiadne |
| **P2-5** | Deterministické centroid ID zo `zone_id` | 4 | 🟡 latentné | 1 h | stredné |
| **P2-6** | Reálny `FAIL` v `audit-supply`, percentily namiesto priemerov | 10 | 🟡 nízke | 1 h | nízke |
| **P2-7** | Rozdeliť najväčšie zóny (Brno-střed, -sever) na 2–3 podzóny | 4 | 🟡 nízke | 3 h | vysoké |

---

## P0 — najprv

### P0-1. Pridať chýbajúce brány

**Problém:** 37 % hraničnej dopravy nemá kadiaľ vojsť (viď diagnóza A).

**Zmena** — iba `config/brno/sim.yaml`:

```yaml
zoning:
  external_gateways:
    # II. trieda je v OSM `secondary` — bez tohto ich select_gateway_target_nodes
    # vôbec nenájde (default je len motorway/trunk/primary).
    allowed_link_types: [motorway, motorway_link, trunk, trunk_link,
                         primary, primary_link, secondary, secondary_link]
    whitelist:
      - "D1"
      - {ref: "D2", anchor_latlon: [49.123847, 16.644072]}
      - "I/43"
      - {ref: "I/52", anchor_latlon: [49.122980, 16.604120]}
      - "I/50"        # východ, 24 271 voz/deň
      - "II/602"      # západ (Jihlavská), 18 523
      - "II/380"      # juhovýchod, 12 348
      - "II/430"      # východ, 10 115
```

Whitelistové cesty obchádzajú filter `supernetwork.eligible_gateway_types`
(`mask = type_ok | is_whitelist`, [supernetwork/pipeline.py:76](src/sim/supernetwork/pipeline.py#L76)),
takže tam netreba nič meniť. Ak by sa zapol `auto_discover`, treba doplniť aj
`secondary` do `eligible_gateway_types`.

Prvé štyri pridané brány pokryjú **65 052 z 137 093** chýbajúcich voz/deň (47 %),
s 15286 a 6401 to je 86 232 (63 %).

**Pozor:** `external_local.total_daily_trips` je **pevný rozpočet** (35 334).
Pridanie brán ten rozpočet neprepočíta, len ho rozdelí na viac dverí — takže
samotné pridanie brán zníži objem na D1/D2/I52/I43. To je zámer, ale znamená to,
že P0-1 sa musí vyhodnotiť **spolu** s kontrolou, či celkový objem nespadol
pod pozorovanie.

**Ako zmerať:** holdout metriky proti baseline + objemy na 8 hraničných cestách
z tabuľky v diagnóze A (pomer model/CSD by sa mal posunúť k 1.0).

### P0-2. Prestať vylučovať sčítania podľa toho, ako sa model mýli

**Problém:** viď diagnóza B.

**Zmena v kóde** ([matching.py:38](src/sim/calibration/matching.py#L38)) —
rozdeliť `_excluded` na dva stĺpce:

- `_excluded_geom` — nízka geometrická kvalita, link odpojený od grafu,
  nulový objem na major ceste (dnes riadky 393–412). **Vylučovať naďalej.**
- `_excluded_ratio` — pomer mimo [0.2, 5]. **Nevylučovať z metrík**, len označiť
  a reportovať zvlášť.

**Zmena v configu:** `exclude_csd_roads: []` a `csd_validation_exclude_sil: []`
(po P0-1, keď tie cesty už majú bránu).

**Ako zmerať:** metriky sa **zhoršia** — to je v poriadku, dovtedy boli
nadhodnotené. Dôležité je, o koľko: rozdiel medzi „plná vzorka" a „filtrovaná
vzorka" je číslo, ktoré patrí do práce.

### P0-3. Zjednotiť generáciu segmentu `other`

**Problém:** viď diagnóza C.

**Dve možnosti, treba sa rozhodnúť:**

| | Ako | Dôsledok |
|---|---|---|
| **(a) seed = tvar, P/A = miera** | ponechať, ale **napísať to** do dokumentácie a v `build-demand` logovať, že segmentové parametre určujú len tvar | žiadna zmena čísel, len jasnosť |
| **(b) zjednotiť hodnoty** | `segments.other.{trip_rate,car_share,occupancy}` = `distribution.pa_*` | segment klesne z ~164 tis. na ~93 tis., celkový dopyt −13 % |

Odporúčam **(b)** — mať v modeli dve rôzne čísla pre „koľko necestovných ciest
denne vzniká" je pri obhajobe ťažko udržateľné. A keďže ODME celkový objem aj tak
sťahuje o 28 % nadol, je pravdepodobné, že (b) zároveň zmenší prácu, ktorú musí
kalibrácia urobiť.

**Ako zmerať:** `prior_drift_pct` v `calibration_report.json` → `seed_deviation`.
Ak sa priblíži k nule, seed bol lepší.

### P0-4. Overiť kotvenie brány I/43

**Podozrenie:** CSD dáva I/43 na hranici okresu **40 582** voz/deň, ale screenline
`auto_gw_I43_N` má `observed_total` **22 478** — čo je blízko hodnoty I/43 na
hranici okresu Blansko (24 265), teda ~15 km severnejšie. Ak sa brána páruje
s nesprávnym CSD úsekom, celý severný koridor je kalibrovaný na ~55 % skutočnosti.

**Ako overiť:**
```bash
python run.py --config config/brno/sim.yaml validate
# potom v outputs/brno/baseline/demand/matching_diagnostics.csv
# nájsť riadky s csd_road == "43" a pozrieť _dist, _match_quality, usek
```
Ak sedí podozrenie, riešenie je `anchor_latlon` pre I/43 (rovnako ako pre D2 a I/52)
alebo explicitný `corridor_observed` záznam v `gateway_calibration`.

### P0-5. Opravná linka, ktorá sa stala hlavným ťahom

**Nález:** `repair_divided_highway_dead_ends` vytvára spojky s úmyselne
penalizovanými parametrami (20 km/h, 200 voz/h, 1 pruh —
[connectivity.py:306-308](src/sim/network/connectivity.py#L306-L308)), aby po nich
model nechodil. V behu `updated_version_1` je takých liniek 13; 11 má nulový tok
(fungujú podľa zámeru), ale:

| link_id | dĺžka | objem | V/C | congested time |
|---|---:|---:|---:|---:|
| **117130** | 365 m | **28 178** | **12.92** | **274 535 s (76 h)** |
| 117138 | 66 m | 6 638 | 1.88 | 86 s |
| 117137 | 34 m | 1 376 | 0.63 | 6 s |

Linka 117130 je `motorway`, 365 m dlhá (na „spojku dvoch slepých koncov pár metrov
od seba" príliš dlhá) a nesie 28 tis. voz/deň. V UE by takto drahú linku nikto
nepoužil, pokiaľ neexistuje alternatíva — je to teda **most (cut edge)**, jediné
spojenie dvoch častí siete, a je dimenzovaný na 200 voz/h.

**Dôsledky:**
- `Delay_factor` 4 176× a LOS F na mieste, kde v skutočnosti žiadne zdržanie nie je
  → skreslené výstupy kongescie vo frontende
- **skim matica**: cestovný čas cez tento úsek je 76 h, takže všetky OD dvojice,
  ktoré ním prechádzajú, majú v `skims.aem` nezmyselnú impedanciu → priamo to
  ovplyvňuje kalibráciu β v `distribute`
- v scenároch to funguje ako umelá bariéra, ktorá tlačí trasy na obchádzky

**Riešenie:** zistiť, čo linka 117130 v skutočnosti spája (`a_node`/`b_node`
v projektovej DB), a buď jej dať reálne parametre mainline diaľnice, alebo
opraviť príčinu — dva jazdné pásy, ktoré `remove_disconnected_components_keep_largest`
rozpojil a repair funkcia zlepila penalizovanou spojkou.

**Diagnostika, ktorá to odchytí do budúcna** — do `audit-supply` pridať kontrolu:
*žiadna linka s `capacity == 200` (opravná spojka) nesmie niesť viac než X voz/deň*.

---

## P1 — potom

### P1-1. Náhodný split + viac seedov

`corridor` stratégia dnes zoradí cesty podľa počtu úsekov zostupne a greedy
naplní kalibračnú časť ([observed.py:554-568](src/sim/calibration/observed.py#L554-L568)),
takže najväčšie koridory idú vždy do kalibrácie a holdout dostáva zvyšky.
Zmeniť na náhodný výber v rámci triedy do naplnenia kvóty a spustiť validáciu
pre 5–10 seedov. Výsledok reportovať ako medián + rozsah.

Pri n=14 („thin" holdout) je to jediný spôsob, ako odlíšiť signál od šumu.

### P1-2. Zamestnanosť vrátane pracujúcich v mieste

`derive_zone_employment` vyhadzuje riadky `lokalizace = 0_na_adrese_OP`
([employment.py:96-99](src/sim/datasets/employment.py#L96-L99)), takže obec,
kde ľudia pracujú doma, vyjde ako miesto bez pracovných miest. Atrakcie v IPF sú
priamo úmerné tomuto číslu.

**Riziko zmeny: stredné** — zmení priestorové rozloženie segmentu `other`,
teda ~1/3 matice. Robiť až po P0-3, inak sa efekty pomiešajú.

### P1-3. Párovanie zón cez kód obce

Nahradiť fuzzy kaskádu v [population.py:160-215](src/sim/datasets/population.py#L160-L215)
joinom na `uzemi_kod`. Zruší:
- mediánový fallback (dnes vymyslená populácia pre nespárované zóny)
- deduplikáciu rovnomenných obcí „ber najväčšiu"
- nerenormalizovaný strop `max_zone_mc_share = 0.70`

### P1-4. Odstrániť tiché fallbacky

[gravity.py:46](src/sim/distribution/gravity.py#L46) a
[gravity.py:93](src/sim/distribution/gravity.py#L93) — `except Exception: pass`.
Nahradiť `logger.warning(..., exc_info=True)` a zapísať použitú vetvu do reportu.
Dnes nevieš, či β počítala AequilibraE alebo hrubá `polyfit` regresia bez
členov produkcie a atrakcie.

### P1-5. Vážené denné faktory

[temporal.py:90-91](src/sim/demand/temporal.py#L90-L91):
`mean(ipd_o / o)` → `sum(ipd_o) / sum(o)`.

### P1-6. Nákladná doprava do tranzitu

Dnes je tranzit odvodený **výhradne z dochádzky za prácou a školou** a
kompenzovaný faktorom `through_traffic_scale = 0.30`. CSD má stĺpce
`tv`/`t`/`pn`/`tn` (ťažké vozidlá), takže podiel nákladnej dopravy na bránových
profiloch je známy. Zaviesť samostatný nákladný segment s vlastnou `pce`
a `through_traffic_scale` prestane byť čiernou skrinkou.

---

## P2 — keď bude čas

Detaily a odkazy do kódu sú v [pipeline_detail.md](pipeline_detail.md).
Sú to prevažne latentné veci — dnes nepadajú, ale raz za čas ticho pokazia
výsledok. Oprav ich pri najbližšom dotyku daného súboru.

Výnimka: **P2-7 (rozdelenie veľkých zón)** má vysoké riziko zmeny (prečísluje
centroidy, zneplatní matice a skimy), ale je to jediné zlepšenie priestorového
rozlíšenia, ktoré nepotrebuje nové dáta. Ak sa doňho pustíš, tak až po tom, čo
budú P0 hotové a zmerané.

---

## Rola kamdojizdime

### Čo dáta vedia a čo nevedia

| | |
|---|---|
| ✅ objem na obec, 4 sezóny, 6 typov ciest | ❌ **trasy** — žiadny smer vstupu, žiadna cesta |
| ✅ denný chod po hodinách × 7 dní | ❌ vnútromestské toky (rozlíšenie = celá obec) |
| ✅ sezónnosť (T1 v lete −28 %) | ❌ 61–64 % relácií je cenzurovaných (prah 7 osôb) |
| ✅ day-type faktory z prítomnosti nerezidentov | ❌ `tranzitujici` nie je tranzitná doprava |

**Kľúčové:** kamdojizdime **samo o sebe brány neurčí**, lebo neobsahuje trasy.
Ale v kombinácii s tým, čo pipeline už má (supernetwork počíta cenu trasy
z každej obce do každej brány → `external_gateway_lookup.parquet`), sa z objemov
na obec dá odvodiť, ktorou bránou tá obec reálne vchádza.

### Kde to najviac pomôže — zoradené

**1. Priestorové rozdelenie `external_local` (najväčší prínos)**

Dnes: pevných 35 334 voz/deň sa medzi brány delí váhou podľa **typu cesty** a
medzi interné zóny podľa **populácie** ([seeds.py:93-99](src/sim/demand/seeds.py#L93-L99))
— priestorovo úplne slepé.

Po zmene: objem na obec × smerovanie zo supernetworku → každá brána dostane
svoj skutočný podiel. To je presne tá „analýza ciest", ktorú model potrebuje.

**Ale až po P0-1.** Smerovať kamdojizdime dáta na 5 brán by len presnejšie
doručilo autá na nesprávne miesta. Pozri, odkiaľ TOP obce reálne vchádzajú
(T1, leto 22):

| smer | obce | cesta | brána dnes |
|---|---|---|---|
| západ/JZ | Střelice 1 135, Troubsko 952, Rosice 1 087, Ivančice 878, Popůvky 685 | **II/602** | ❌ |
| východ | Šlapanice 2 082, Mokrá-Horákov 903, Slavkov 753, Rousínov 697 | **I/50, II/430, 15286** | ❌ |
| SV (Svitava) | Bílovice n. Svit. 1 448, Adamov 795, Blansko 1 573 | **II/383, II/374** | ❌ |
| juh | Moravany 1 491, Rajhrad 877 | 15275/15276, II/152 | čiastočne |
| sever | Kuřim 2 255, Lelekovice 668, Tišnov 969 | I/43, **6401** | čiastočne |

**2. Tretia validačná vrstva**

Model dnes validuje len proti CSD (objemy na linkách). Kamdojizdime dáva
**nezávislý** pohľad: OD objemy na obec a bránové toky. To je metodicky silné —
validácia proti inému typu dát, nie proti tomu istému zdroju z iného úseku.

**3. Sezónnosť a day-type faktory**

Jediné dáta v projekte, ktoré kvantifikujú sezónnosť (T1 v lete −28 %, ale
celkový objem nezmenený — presun z dochádzky do návštevníkov). A day-type
faktory z prítomnosti nerezidentov (streda 1.17, sobota 0.51, nedeľa 0.42) sú
priamo porovnateľné s tým, čo dnes počíta `learn-profile` z CSD — ideálna
krížová kontrola.

**4. Rozdelenie do období dňa**

`denni_chod` ukazuje, že ranná špička je ostrejšia než popoludňajšia
(najsilnejší nárast 07–08 h +19 571, najsilnejší pokles 16–17 h −13 183),
zatiaľ čo `defaults.py` má `am` a `pm` zhruba symetrické.

### Čo z kamdojizdime nerobiť

- **Nenahrádzať SLDB.** 61–64 % relácií je cenzurovaných (prázdna bunka ≠ nula),
  takže OD matica je useknutá zľava.
- **Nepoužívať `tranzitujici`** ako tranzitnú dopravu — je to iná veličina.
- **Neodvodzovať vnútromestské toky** — rozlíšenie je celá obec.

---

## Navrhované poradie prác

Každý krok má merateľný výstup a dá sa zastaviť.

```
0. Zamraziť baseline            → outputs_baseline_v0 (už existuje)
   ↓
1. P0-4  overiť bránu I/43      → matching_diagnostics.csv, 2 h
   ↓
2. P0-1  pridať 4–6 brán        → holdout metriky + tabuľka model/CSD na hranici
   ↓
3. P0-2  prestať filtrovať      → metriky na plnej aj filtrovanej vzorke
         podľa zhody              (očakávaj zhoršenie — to je správne)
   ↓
4. P0-3  zjednotiť `other`      → prior_drift_pct v calibration_report
   ↓
5. P1-1  náhodný split, 5 seedov → medián + rozsah metrík (koniec „thin holdout" hádania)
   ↓
6. kamdojizdime fáza 1 (YAML)   → total_daily_trips ako citlivosť
   ↓
7. kamdojizdime fáza 2 (loader) → external_local smerovaný na brány
   ↓
8. P1-2..P1-6, P2-*
```

**Po každom kroku odložiť:** `validation_report.json`, `calibration_report.json`,
`assignment_results.parquet`, `od_summary.json`.

**Pozor na „thin" holdout (n=14):** rozdiel 0.02 v R² medzi behmi nie je signál.
Až po kroku 5 (viac seedov) sa dá o zlepšeniach hovoriť s istotou.

---

## Čo nerobiť

- **Neladiť `through_traffic_scale` ani `car_share`, kým platí problém A.**
  Ladenie globálnych násobičov na modeli, ktorý má dopravu na nesprávnej strane
  mesta, je hľadanie kompenzácie za štrukturálnu chybu.
- **Nepridávať scenárom elasticitu dopytu** — je to veľa práce a model ju
  neuvezie (jedno denné číslo, 69 zón). Lepšie je limitáciu jasne priznať.
- **Neinvestovať do výkonu.** Beh trvá minúty a nie je to predmetom práce.
- **Nemeniť naraz viac vecí.** Pri n=14 sa efekty nedajú rozpliesť.
