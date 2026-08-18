# Kamdojizdime.cz — plán

Jediný pracovný súbor pre túto tému.

**Stav k 15. 8. 2026:** dáta preskúmané priamo z CSV (nie z notebooku — ten je
starší a časť jeho záverov neplatí, viď [Opravy](#opravy-voči-staršej-verzii-plánu)).
Do pipeline sa zatiaľ nič neimplementovalo.

Práca je rozdelená na tri fázy, ktoré sa dajú robiť nezávisle a každá má vlastný
merateľný výstup:

| Fáza | Čo | Zmena v repe | Výstup |
|---|---|---|---|
| **0** | Baseline — odložiť referenčné metriky | žiadna | validačné metriky pred zmenou |
| **1** | Úprava parametrov z týchto dát | iba YAML config | metriky po zmene, porovnanie s fázou 0 |
| **2** | Zakomponovanie do pipeline | nový loader + demand modul | OD matica z mobilných dát |

---

## Dáta

Mobilné geolokačné dáta pre Brno (obec `582786`), **všetky 4 sezóny**:
`podzim_21`, `jaro_22`, `leto_22`, `zima_22_23` v `data/kam_dojizdime_*`.

| Súbor | Obsah | Riadkov / sezónu |
|---|---|---|
| `bydlici` | rezidenti Brna podľa kategórií (Tabuľka č. 1) | 1 |
| `dojizdka` / `vyjizdka` | OD matica Brno ↔ ~5 700 obcí (Tabuľka č. 2) | 23 000 / 24 000 |
| `denni_chod` | 24 h × 7 dní prítomnosť v Brne (Tabuľka č. 3) | 168 |

Zdroj pravdy pre výklad stĺpcov: metodika ŘSD/INTENS
`data/kam_dojizdime_zima_22_23/27_metodicky-postup-reseni-zima-2022-2023.pdf`.

Formát CSV: oddeľovač `;`, desatinná bodka, BOM (`utf-8-sig`).

### Výklad `typ_cesty` (metodika §7.2, Tabuľka č. 2, stĺpec E)

| # | Kód | Význam | Kritérium |
|---|---|---|---|
| 1 | T1 | dojížďka za prací/školou | ≥13 pobytov a ≥50 h za 28 dní |
| 2 | T2 | intenzivní dojížďka za službami | ≥4 pobyty a ≥8 h |
| 3 | T3 | občasná dojížďka za službami | ≥2 pobyty a ≥4 h |
| 4 | D2 | druhé bydlení | ≥6 prenocovaní; **nesčítava sa** (už je v T2/T3) |
| 5 | PN | přenocující návštěvník | ≥1 prenocovanie |
| 6 | N | návštěvník | jeden pobyt ≥3 h |

Dve veci, na ktorých sa dá ľahko pomýliť:

- **`pocet_osob` nie sú cesty.** Je to počet *osôb s príznakom za 28-dňové obdobie*.
  Prevod na cesty/deň je náš predpoklad, nie údaj z dát.
- **`pocet_osob_pd` = „pracovná doba"** (Po–Pá 6–18), nie „pracovný deň".
  Pre D2 a PN sa neurčuje → v dátach 0.

### Obmedzenia zdroja

- **Cenzúra: 61–64 % relácií je potlačených.** Prah zverejnenia je 7 osôb
  (metodika §3.5); v exporte je potlačená hodnota **prázdna bunka**, nie `"xx"`.
  Prázdne ≠ nula. → OD matica je useknutá zľava, **nemôže SLDB nahradiť úplne**.
- **Rozlíšenie = celá obec.** Vhodné na gateway/externé toky, nie na vnútromestské linky.
- **`denni_chod` je prítomnosť, nie tok.** Kto je práve na ceste, spadne do internej
  kategórie „na cestě", ktorá sa **neexportuje**. Špička v prítomnosti je preto
  posunutá a vyhladená oproti špičke v doprave.
- **`tranzitujici` NIE JE tranzitná doprava** — viď [nižšie](#čo-sa-použiť-nedá).

---

## Čo dáta ukazujú

Overené priamo z CSV (skript v scratchpade, výsledky nižšie sú z `zima_22_23`,
ak nie je uvedené inak).

### Cenzúra po sezónach

| sezóna | relácií spolu | prázdnych | podiel |
|---|---|---|---|
| podzim_21 | 48 146 | 30 243 | 62.8 % |
| jaro_22 | 48 901 | 30 594 | 62.6 % |
| leto_22 | 49 880 | 30 301 | 60.7 % |
| zima_22_23 | 47 212 | 30 340 | 64.3 % |

### Objemy podľa typu cesty (osoby / 28 dní)

| typ_cesty | do Brna | z Brna | relácií (do/z) |
|---|---|---|---|
| T1 práca/škola | 97 290 | 26 514 | 795 / 483 |
| T2 intenzívna | 127 268 | 69 734 | 1 477 / 1 476 |
| T3 občasná | 138 039 | 119 022 | 1 941 / 2 001 |
| D2 druhé bývanie | 8 596 | 10 374 | 339 / 435 |
| PN prenocujúci | 40 730 | 99 180 | 897 / 2 036 |
| N návštevník | 142 651 | 137 078 | 2 685 / 2 307 |

Asymetria T1 (97 tis. dnu vs. 27 tis. von) je vecne správna — Brno je
zamestnávateľské centrum.

### Sezónnosť — najsilnejšie nevyužité zistenie

| sezóna | T1 spolu (oba smery) | vs. jeseň | T1+T2+T3+N spolu |
|---|---|---|---|
| podzim_21 | 137 124 | — | 963 231 |
| jaro_22 | 123 116 | −10.2 % | 991 059 |
| **leto_22** | **98 174** | **−28.4 %** | 993 237 |
| zima_22_23 | 123 804 | −9.7 % | 857 596 |

Letný prepad pravidelnej dochádzky o 28 % je veľký. **Model dnes nemá žiadny
sezónny rozmer** — CSD 2025 je celoročný priemer a pipeline produkuje jedno denné
číslo. Toto sú prvé dáta v projekte, ktoré sezónnosť kvantifikujú. Zároveň si
všimnite, že celkový objem (T1+T2+T3+N) v lete **neklesol** — presunul sa z
dochádzky do návštevníkov. Segmenty sa teda musia škálovať oddelene, nie spoločným
faktorom.

### Koncentrácia — agregácia na brány je bezstratová

Obcí s nenulovým T1: **3 247**.

| podiel objemu T1 | koľko obcí |
|---|---|
| 50 % | **47** |
| 80 % | 175 |
| 95 % | **443** |
| 99 % | 703 |

TOP 15: Modřice 5 049, Kuřim 4 340, Šlapanice 3 611, Moravany 2 931, Blansko 2 577,
Bílovice n. Svit. 2 457, Rajhrad 1 808, Troubsko 1 799, Rosice 1 745, Vyškov 1 543,
Tišnov 1 538, Střelice 1 533, Praha 1 447, Mokrá-Horákov 1 371, Ivančice 1 335.

→ Cenzurovaný chvost je pre gateway toky nepodstatný.

### Denný chod (pondelok, `dojizdejici_prace`)

```
04:00-05:00     7,542   d=   +918
05:00-06:00    14,987   d= +7,445
06:00-07:00    31,720   d=+16,733
07:00-08:00    51,291   d=+19,571   ← najsilnejší nárast
08:00-09:00    63,423   d=+12,133
11:00-12:00    70,239   ← peak prítomnosti
16:00-17:00    41,396   d=-13,183   ← najsilnejší pokles
```

Amplitúda prítomných spolu: **pondelok 21.9 %**, sobota **3.4 %**, nedeľa 10.2 %.
Ranná špička je ostrejšia než popoludňajšia — príchody sú koncentrovanejšie než
odchody. V modeli sa to dnes nemodeluje (`am` a `pm` share v `defaults.py` sú
zhruba symetrické).

### Day-type faktory z prítomnosti **nerezidentov**

Priemer `dojizdejici_prace + dojizdejici_intenzivne + dojizdejici_obcasne + navstevnici`:

| deň | priemer | faktor vs. pondelok |
|---|---|---|
| pondelok | 51 470 | 1.000 |
| utorok | 55 396 | 1.076 |
| **streda** | 60 173 | **1.169** |
| štvrtok | 49 550 | 0.963 |
| piatok | 43 788 | 0.851 |
| **sobota** | 26 288 | **0.511** |
| **nedeľa** | 21 676 | **0.421** |

### Prevod osôb na vozidlá

Pri `demand.conversion.work`: `car_share 0.48`, `occupancy 1.20`,
`trips_per_person 2.0`; frekvencia = **minimálny** počet pobytov z metodiky:

| typ | osôb (oba smery) | min. pobytov / 28 dní | voz-ciest/deň |
|---|---|---|---|
| T2 intenzívna | 197 002 | 4 | 22 515 |
| T3 občasná | 257 061 | 2 | 14 689 |
| N návštevník | 279 728 | 1 | 7 992 |
| **spolu (= `external_local`)** | | | **45 196** |
| T1 práca/škola | 123 804 | 13 | 45 984 |
| T1 pri plnej dennej dochádzke | 123 804 | 20 (Po–Pá) | **99 043** |

---

## Fáza 0 — Baseline

Bez baseline sa nedá ukázať, či zmena parametrov pomohla.

**Baseline už existuje** v `outputs_pred/brno/baseline/` (beh z 31. 7. 2026).
Odporúčam premenovať na niečo jednoznačné a nechať tak:

```bash
mv outputs_pred outputs_baseline_v0
```

Referenčné čísla z toho behu
(`outputs_baseline_v0/brno/baseline/demand/validation_report.json`):

| metrika | kalibračná časť (65 %) | **holdout (35 %)** | cieľ |
|---|---|---|---|
| R² | 0.839 | **0.700** | ≥ 0.80 |
| slope | 0.932 | **0.715** | 0.85–1.15 |
| %RMSE | 38.9 | **42.2** | ≤ 35 |
| bias | −14.0 % | **−18.9 %** | ≤ 15 % |
| max. odchýlka screenline | — | 58.4 % | ≤ 15 % |
| `holdout_adequacy` | — | **"thin"** (14 bodov) | — |
| `overall_pass` | false | **false** | — |

Verdikt sa berie z holdoutu (`verdict_source: "holdout"`).

**Súbory, ktoré treba odložiť pri každom behu:**
`validation_report.json`, `calibration_report.json`, `pre_odme_convergence.json`,
`assignment_results.parquet`.

> **Pozor na „thin" holdout.** 14 bodov je málo; rozdiel 0.02 v R² medzi behmi
> nemusí byť signál. Pri porovnávaní verzií uvádzajte aj `n`.

---

## Fáza 1 — Úprava parametrov (bez zmeny kódu)

Iba zmena hodnôt v YAML. Žiadny nový loader, žiadny nový modul.

**Všetky zmeny idú do [config/brno/sim.yaml](config/brno/sim.yaml)**, nie do
`defaults.py` — defaults sú spoločné pre všetky mestá.

### 1a. `external_local.total_daily_trips`

**Kde:** [config/brno/sim.yaml:45](config/brno/sim.yaml#L45), dnes `130000`.

Segment `external_local` predstavuje **cesty medzi bránami a mestom, ktoré nie sú
pravidelná dochádzka** — teda T2 + T3 + N. Pravidelná dochádzka (T1) ide cez
`commuting` a škáluje sa parametrom v bode 1b.

Metodika dáva **minimálny počet pobytov** pre každý typ, čo je tvrdá dolná hranica
frekvencie — netreba hádať:

| Typ | Osôb | Min. pobytov / 28 dní | Voz-ciest/deň |
|---|---|---|---|
| T2 intenzívna | 197 002 | 4 | 22 515 |
| T3 občasná | 257 061 | 2 | 14 689 |
| N návštevník | 279 728 | 1 | 7 992 |
| **spolu** | | | **45 196** |

**45 196 je podlaha, nie odhad** — skutočná hodnota je vyššia, lebo kritériá sú
minimá. Dnešných 130 000 zodpovedá ~6.2 pobytom z 28 dní naprieč typmi, čo je nad
minimom pri všetkých troch typoch, ale nie je to podložené.

**Odporúčanie:** dáta nedávajú bodový odhad, dávajú interval. Spustiť ako
citlivosť štyri hodnoty a nechať rozhodnúť holdout validácia:

```yaml
demand:
  segments:
    external_local:
      total_daily_trips: 45000    # podlaha z metodiky
      # total_daily_trips: 90000  # 2× minimálna frekvencia
      # total_daily_trips: 130000 # dnešná hodnota
      # total_daily_trips: "auto" # nezávislý odhad z CSD
```

`"auto"` počíta pipeline z CSD AADT na gateway cestách
([seeds.py:132](src/sim/demand/seeds.py#L132)) — je to **štvrtý nezávislý odhad**
a oplatí sa ho spustiť pre porovnanie.

> ⚠️ **Nepoužívať 91 000.** To je súčet vrátane T1 (45 984) — T1 patrí do
> `commuting`, nie do `external_local`. Použitie 91 000 by pravidelnú dochádzku
> započítalo dvakrát.

### 1b. `external_commuting_scale`

**Kde:** [config/brno/sim.yaml:48](config/brno/sim.yaml#L48), dopísať ako súrodenca
`through_traffic_scale`. Dnes sa berie default `0.65` z
[defaults.py:219](src/sim/defaults.py#L219).

Tento parameter **nie je absolútne číslo, ale pomer** — násobí objem externej
dojížďky odvodenej zo SLDB
([od_builder.py:211-213](src/sim/demand/od_builder.py#L211-L213)). Nová hodnota sa
preto nedá napísať bez toho, aby sa vedelo, koľko ten SLDB objem je:

```
nová hodnota = 99 043 / (SLDB externá dojížďka pri scale 1.0)
```

Čitateľ **99 043 voz-ciest/deň** je z T1 (oba smery, plná denná dochádzka).
Menovateľa dá baseline. Postup (`build-demand` je rýchly, netreba celý pipeline):

1. `build-demand` s `external_commuting_scale: 1.0` → odčítať
   `final_cores_sum["wd_daily_commuting"]` z
   `outputs/brno/baseline/demand/od_summary.json` = **C₁**
2. to isté s `0.65` = **C₂**
3. `SLDB externá dojížďka = (C₁ − C₂) / 0.35`
4. `nová hodnota = 99 043 / výsledok kroku 3`

### 1c. Časové rozdelenie — spočítané, ale **zatiaľ NEMENIŤ**

Z `denni_chod` sa dá pre `external_local` odvodiť rozdelenie z prítomnosti T2+T3+N:

| | am | ip | pm | ev |
|---|---|---|---|---|
| odvodené z dát | 0.43 | 0.15 | 0.23 | 0.19 |
| dnes v `defaults.py` | 0.28 | 0.22 | 0.30 | 0.20 |

**Neaplikovať.** Odvodenie stojí na hodinovej zmene prítomnosti, ktorá je **netto**
(príchody mínus odchody v tej istej hodine), takže systematicky podhodnocuje `ip`
a nadhodnocuje `am` — smer skreslenia je známy. Ranná špička 0.43 pri segmente,
ktorý predstavuje *služby a návštevníkov*, je aj vecne nepravdepodobná. Patrí to do
fázy 2, kde sa dá rozklad urobiť poriadne (rozdeliť príchody a odchody, nie brať
ich rozdiel).

### 1d. Day-type faktory — **použiteľné ako krížová kontrola**

Faktory z prítomnosti nerezidentov (streda 1.169, sobota 0.511, nedeľa 0.421) sa
dajú porovnať s tým, čo učí `learn-profile` z CSD
([temporal.py](src/sim/demand/temporal.py)). Zhoda = nezávislé potvrdenie;
nezhoda = viete, ktorý zdroj preveriť.

Toto **nevyžaduje spustenie pipeline** — stačí porovnať s existujúcim
`temporal_profile.json`. Najlacnejšia položka celej fázy 1.

### Prepočet

Sieť sa nemení, takže sa **neopakuje** `build-network`, `fetch-data`,
`normalize-network`, `build-zones` ani `build-supernetwork`:

```bash
CFG=config/brno/sim.yaml
python run.py --config $CFG build-demand
python run.py --config $CFG assign-warm-skims   # regeneruje sa sám, OD matica je novšia
python run.py --config $CFG distribute
python run.py --config $CFG assign
python run.py --config $CFG audit-supply
python run.py --config $CFG calibrate
python run.py --config $CFG validate
```

### Ako porovnávať — tri úrovne

**Zmeniť naraz iba jeden parameter.** A porovnávať na troch úrovniach, lebo ODME
časť rozdielu „zožerie" (gradientne dofituje aj horší vstup):

| Úroveň | Čo | Prečo |
|---|---|---|
| **1. pred ODME** | `assignment_results.parquet` + `pre_odme_convergence.json` po kroku `assign` | čistý prínos dát |
| **2. po ODME** | `benchmarks.holdout_validation` z `validation_report.json` | prínos v praxi |
| **3. deviácia od seedu** | `seed_deviation.prior_drift_pct` a `n_cells_exceeding_threshold` v `calibration_report.json` | **najsilnejší argument** — lepší vstup ⇒ ODME musí maticu pokriviť *menej* |

Tretia úroveň je metodologicky najzaujímavejšia a v diplomovke sa píše sama.

**Výstup fázy:** tabuľka baseline vs. upravené parametre na metrikách z
`validation_report.json`.

---

## Fáza 2 — Zakomponovanie do pipeline

Až keď fáza 1 ukáže, že zdroj dáva zmysel.

**Čo treba napísať:**

1. **Loader** `src/sim/datasets/kamdojizdime.py` — načítanie 4 sezón, cenzúra ako
   samostatný príznak (nie nula), výklad `typ_cesty` z tabuľky vyššie.
   Píše sa a testuje **bez aequilibrae** → najrýchlejšia slučka, začať tu.
2. **Mapovanie obec → gateway.** Pipeline už má na externé toky mašinériu pre SLDB
   (`external_gateway_lookup.parquet`, supernetwork through-pairs). Cieľ je poslať
   kamdojizdime OD cez tú istú cestu, nie stavať novú.
3. **Napojenie do `build-demand`**
   ([od_builder.py:159](src/sim/demand/od_builder.py#L159)) — blend so SLDB.
4. **Sezónny prepínač** — ktorá sezóna je baseline (CSD 2025 je celoročný priemer).
   Vzhľadom na letný prepad T1 o 28 % to nie je kozmetika: sezóna sa musí voliť
   vedome a **oddelene pre commuting a pre návštevnícke segmenty**.

**Replace vs. blend:** kvôli ~63 % cenzúre je odporúčanie **blend** — SLDB drží
štruktúru a dlhý chvost malých obcí, kamdojizdime kalibruje objem. Čisté nahradenie
by zahodilo relácie pod prahom 7 osôb.

---

## Čo sa použiť NEDÁ

**`tranzitujici` na `through_traffic_scale`.** Metodika §7.3, stĺpec L to definuje
ako *neklasifikovaný pobyt ≥30 min bez príznaku voči obci* — nie prejazd. Kto Brnom
naozaj prejde bez zastávky, spadne do neexportovanej kategórie „na cestě".
Dáta to potvrdzujú: 0.4–0.6 % prítomných osôb, čo na tranzit cez Brno nesedí ani
rádovo.

→ Tretia validačná vrstva teda stojí na **OD objemoch a gateway tokoch**, nie na
tranzite. Tranzit ostáva na supernetwork + CSD.

---

## Validácia — čo kamdojizdime pridáva

CSD (linkové sčítanie) je dnes **jediná** vrstva, ktorá vstupuje do
`validation_report.json`. Waze je len v `experiments/` a do reportu nevstupuje.

Kamdojizdime by bola tretia, nezávislá vrstva — osoby/OD namiesto vozidiel na úseku:

- súčty `dojizdka`/`vyjizdka` vs. modelové **gateway objemy**
- day-type faktory vs. CSD-učený `temporal_profile.json` (fáza 1d)
- **sezónna validácia** — 4 obdobia sú k dispozícii hneď, model dnes sezónnosť nemá

---

## Súvisiaci blocker: Waze / Postgres je nedostupný

Netýka sa kamdojizdime priamo, ale blokuje druhú validačnú vrstvu a treba to
vyriešiť skôr, než sa bude porovnávať čokoľvek „pred vs. po".

`data/brno/cache/datasets_manifest.json` hlási pre všetky štyri Postgres zdroje
(`closures_pg`, `traffic_jams_pg`, `road_segments_pg`, `event_links_pg`):

```
could not translate host name "REDACTED_HOST" to address
```

Default v [defaults.py:507](src/sim/defaults.py#L507) má redigovaný host a prázdne
heslo. `data/brno/cache/` teda neobsahuje `jams.parquet`,
`jams_segment_stats.parquet`, `road_segments.parquet` ani `closures.parquet` →
`exp06`, `exp07`, `exp10`, `exp11` spadnú na `FileNotFoundError`.

Riešenie: doplniť `closures_db` do konfigurácie (mimo gitu) a spustiť
`fetch-data --force --only traffic_jams_pg road_segments_pg closures_pg`.

---

## Článok (Informatics 2026, IEEE, Poprad)

*Open-source, automatizovaná pipeline pre What-If analýzu dopravných uzávierok;
overená na 3 mestách (Brno/Olomouc/Most); validácia cez ŘSD + Waze. Základ:
Kaňkovský DP, dátová platforma Ondrušková 2024.*

Kam pridať kamdojizdime prácu (samostatná, merateľná časť nad rámec DP):

- §1 Contributions: validácia z 2 → 3 nezávislých zdrojov
- §3.3: nová podčasť — external segment z mobilných dát namiesto fudge faktora
- §3.3/§3.4: `denni_chod` → napĺňa existujúcu vetu v §7 Conclusion o AM/PM špičkách
- §4/§6: nová validačná tabuľka (gateway objemy, sezónnosť)
- limitácia na priznanie: dáta len pre Brno → patrí do §4, nie §5 (multi-city)
- limitácia na priznanie: ~63 % relácií pod prahom anonymizácie
- limitácia na priznanie: holdout je „thin" (14 bodov)

Riziká: Olomouc slabší bias/menej kalibračných bodov (§5), Waze koreláciu vždy
podať s kontextom, citácie v §2/§6 overiť.

---

## Opravy voči staršej verzii plánu

Staršia verzia stála na notebooku
[notebooks/kamdojizdime_explorace.ipynb](notebooks/kamdojizdime_explorace.ipynb).
Tri veci sa pri kontrole priamo z CSV ukázali inak:

1. **Day-type faktory NIE SÚ ploché.** Staré tvrdenie „sobota 0.87, nedeľa 0.89 →
   na náhradu nie" vychádzalo z `pritomni_celkem`, kde 350 tis. rezidentov
   prehluší všetko ostatné. Počítané **len z nerezidentov** vychádza sobota
   **0.511**, nedeľa **0.421**, streda **1.169**. To je ostré rozdelenie a je
   použiteľné (fáza 1d).
2. **Sezónnosť bola podcenená.** Stará verzia ju spomínala len ako jednu odrážku vo
   validácii. Reálny letný prepad T1 je **−28.4 %** a pritom celkový objem
   neklesol — segmenty sa musia škálovať oddelene. To má priamy dopad na návrh
   fázy 2.
3. **Koncentrácia:** 95 % objemu T1 nesie **443** obcí (staré: ~424). Rozdiel je
   metodický (sezóna, spôsob agregácie oboch smerov), záver — agregácia na brány je
   bezstratová — platí.

Ostatné závery staršej verzie sa potvrdili: výklad `typ_cesty`, hodnota
99 043 voz-ciest/deň pre T1, podlaha 45 196 pre `external_local`, nepoužiteľnosť
`tranzitujici`, odporúčanie blend namiesto replace.

---

## Otvorené otázky

1. **Replace vs. blend** externého segmentu — odporúčanie je blend, rozhodnutie
   nie je urobené.
2. `data.brno.cz` — overiť, či existuje mimo tohto repa, alebo opraviť v článku.
3. Dáta len pre Brno — potrebné pre Most/Olomouc, alebo obmedziť scope na §4.
4. `SRCH-37188_zoznam_rus.csv` (iba v `zima_22_23`) — neidentifikovaný súbor,
   žiadny prienik s kódmi obcí, nie je súčasťou troch tabuliek metodiky.
   Zistiť pôvod u dodávateľa.
5. Prístupové údaje k Postgresu (blocker vyššie).
