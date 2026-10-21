# Emoji Multi-Attractor – Modellübersicht

Quelle: `models/bundles/nca-emoji/emoji-multi-attractor-outsource` (normal) und `...-outsource-v2` (v2).
Status aus `manifest.yaml` (`status`, `training.completed`, `training.error_code`). Fehlercodes laut `NCA/trainer/runner.py::_divergence_code`:

- 1 = Loss nicht endlich (NaN/Inf)
- 2 = Zustände nicht endlich (NaN/Inf in den States)
- 3 = Loss explodiert (|loss| > 1e16)

**Hinweis:** Der exakte Abbruch-Schritt wird nur in W&B geloggt (`Training stopped with error X at step Y`), nicht im Manifest. `letzter OK-Checkpoint` = beste gespeicherte Iteration; der Abbruch kam danach. Alle Läufe waren auf 16000 Iterationen geplant. Alle `model.eqx` stimmen mit dem SHA256 im Manifest überein.

Ordnerstruktur: `<version>/<NCA|gNCA>/<circular|replicate>/<N>_attraktoren/<kombination>/<funktioniert|abgebrochen>/<run-id>/`

## normal (211 Läufe: 101 funktioniert, 110 abgebrochen)

| Familie | Padding | funktioniert | abgebrochen | davon Err1 | Err2 | Err3 |
|---|---|---|---|---|---|---|
| NCA | circular | 3 | 50 | 15 | 3 | 32 |
| NCA | replicate | 3 | 51 | 12 | 5 | 34 |
| gNCA | circular | 44 | 8 | 4 | 0 | 4 |
| gNCA | replicate | 51 | 1 | 1 | 0 | 0 |

### Abgebrochene Läufe

| Familie | Padding | Attraktoren | Repeat | Fehler | letzter OK-Checkpoint | W&B | Run-ID |
|---|---|---|---|---|---|---|---|
| NCA | circular | alien_monster+crab | 0 | 1 – Loss nicht endlich (NaN/Inf) | 424 / 16000 | bu1vn3a0 | 20261010T121815Z-cfg0c7038571190-b0272d6b |
| NCA | circular | alien_monster+crab | 1 | 3 – Loss explodiert (|loss| > 1e16) | 1253 / 16000 | eaw9hhgu | 20261012T052747Z-cfg067d79f8ffcf-0a048337 |
| NCA | circular | alien_monster+lizard | 0 | 3 – Loss explodiert (|loss| > 1e16) | 354 / 16000 | 712uteul | 20261010T224718Z-cfg00544532b78e-f0e533ea |
| NCA | circular | alien_monster+lizard | 1 | 1 – Loss nicht endlich (NaN/Inf) | 644 / 16000 | thjj0w7q | 20261010T223743Z-cfgc46a64721078-964009c5 |
| NCA | circular | alien_monster+microbe | 0 | 2 – Zustaende nicht endlich (NaN/Inf in states) | 465 / 16000 | 0be7gwzr | 20261012T134515Z-cfg480a3ea45461-2fb340cf |
| NCA | circular | alien_monster+microbe | 1 | 3 – Loss explodiert (|loss| > 1e16) | 662 / 16000 | a8l8jb9s | 20261012T160944Z-cfg430979e7a7db-bb4b4692 |
| NCA | circular | alien_monster+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 820 / 16000 | ae1eg11o | 20261009T174911Z-cfgcb08495688bb-65e40761 |
| NCA | circular | alien_monster+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 2271 / 16000 | tzohms8o | 20261012T102109Z-cfgabf7365ac644-5996ffce |
| NCA | circular | crab+lizard | 0 | 3 – Loss explodiert (|loss| > 1e16) | 541 / 16000 | ump0oprh | 20261009T201218Z-cfg50e9edecb0e2-bd892b4f |
| NCA | circular | crab+lizard | 1 | 1 – Loss nicht endlich (NaN/Inf) | 10013 / 16000 | 0j5gkavo | 20261011T022530Z-cfg23516cc3cd71-0c23cb1a |
| NCA | circular | crab+microbe | 0 | 3 – Loss explodiert (|loss| > 1e16) | 1732 / 16000 | 6mszjgc1 | 20261012T043215Z-cfg48fcfb2ea952-541243ad |
| NCA | circular | crab+microbe | 1 | 3 – Loss explodiert (|loss| > 1e16) | 6162 / 16000 | gzblf7t9 | 20261009T181712Z-cfg3a1d37d1957a-13cce654 |
| NCA | circular | crab+rooster | 1 | 2 – Zustaende nicht endlich (NaN/Inf in states) | 3155 / 16000 | zjq2abff | 20261009T215800Z-cfg451952ea8a64-84a6a334 |
| NCA | circular | lizard+microbe | 0 | 3 – Loss explodiert (|loss| > 1e16) | 321 / 16000 | 760yvi4q | 20261011T232741Z-cfg1d34312acc05-06e0467e |
| NCA | circular | lizard+microbe | 1 | 1 – Loss nicht endlich (NaN/Inf) | 229 / 16000 | x1zjr0vt | 20261010T122120Z-cfge525110b4445-509b85cd |
| NCA | circular | lizard+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 1024 / 16000 | wjv878dn | 20261009T201855Z-cfg08708e597fec-28759889 |
| NCA | circular | lizard+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 779 / 16000 | tjd9ps61 | 20261010T152533Z-cfge051d57e1976-46343b17 |
| NCA | circular | microbe+rooster | 0 | 1 – Loss nicht endlich (NaN/Inf) | 646 / 16000 | v0fk0c2w | 20261009T224653Z-cfg5207a1b79130-f37fb35d |
| NCA | circular | microbe+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 828 / 16000 | um8czvdd | 20261011T115041Z-cfg1eee2fefcfa2-a652d1c8 |
| NCA | circular | alien_monster+crab+lizard | 0 | 1 – Loss nicht endlich (NaN/Inf) | 2432 / 16000 | p2yvxgkj | 20261010T194604Z-cfg5b7735319241-84bc9a47 |
| NCA | circular | alien_monster+crab+lizard | 1 | 3 – Loss explodiert (|loss| > 1e16) | 10655 / 16000 | 5h0ovflq | 20261010T025357Z-cfg8c8cf219e20b-b478afa5 |
| NCA | circular | alien_monster+crab+microbe | 0 | 3 – Loss explodiert (|loss| > 1e16) | 4912 / 16000 | ky62j815 | 20261010T143345Z-cfga774a20f633e-c67e5c68 |
| NCA | circular | alien_monster+crab+microbe | 1 | 3 – Loss explodiert (|loss| > 1e16) | 1002 / 16000 | gnc714ty | 20261012T072415Z-cfg671d1facddf9-00191604 |
| NCA | circular | alien_monster+crab+rooster | 0 | 1 – Loss nicht endlich (NaN/Inf) | 1401 / 16000 | gdczq9lb | 20261010T020335Z-cfg8c485948374c-781a4087 |
| NCA | circular | alien_monster+crab+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 369 / 16000 | 5b93thn4 | 20261010T075455Z-cfg1bb8f653bb90-737ffd37 |
| NCA | circular | alien_monster+lizard+microbe | 0 | 3 – Loss explodiert (|loss| > 1e16) | 1262 / 16000 | el6aaqlb | 20261011T143726Z-cfg33353d106523-3f9752c2 |
| NCA | circular | alien_monster+lizard+microbe | 1 | 1 – Loss nicht endlich (NaN/Inf) | 4859 / 16000 | uo1vw07m | 20261012T091725Z-cfgf868d0f61fa4-5e55bc8f |
| NCA | circular | alien_monster+lizard+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 1208 / 16000 | 1xzif6qs | 20261009T142325Z-cfg4ec5b48611ae-7c55639e |
| NCA | circular | alien_monster+lizard+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 4023 / 16000 | ddxn44zt | 20261012T083453Z-cfg7e0e9cfdc10a-295abe09 |
| NCA | circular | alien_monster+lizard+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 5189 / 16000 | a1l96vzv | 20261012T192040Z-cfgc1180dc7f552-54ecba85 |
| NCA | circular | alien_monster+microbe+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 713 / 16000 | 7ydkpnno | 20261012T051251Z-cfg99f33fe6d541-c25a8acf |
| NCA | circular | crab+lizard+microbe | 1 | 1 – Loss nicht endlich (NaN/Inf) | 4648 / 16000 | cicfsmwf | 20261012T015921Z-cfg432211a9d5ee-9ab3f99e |
| NCA | circular | crab+lizard+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 1410 / 16000 | 6qg3efnn | 20261009T211603Z-cfg1978e2fc9ceb-73cfc5a6 |
| NCA | circular | crab+lizard+rooster | 1 | 1 – Loss nicht endlich (NaN/Inf) | 2672 / 16000 | xnfxizi5 | 20261010T061134Z-cfg3ad802362dec-54e7ed09 |
| NCA | circular | crab+microbe+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 1059 / 16000 | qr66f7fr | 20261012T152942Z-cfga97cb27cf44d-abd91738 |
| NCA | circular | crab+microbe+rooster | 1 | 1 – Loss nicht endlich (NaN/Inf) | 1190 / 16000 | mk5hwq7d | 20261010T123407Z-cfg6cf68bff16f0-28d2bc6e |
| NCA | circular | lizard+microbe+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 3177 / 16000 | y5jkivfp | 20261012T104300Z-cfg699ad3b677a1-d2707244 |
| NCA | circular | lizard+microbe+rooster | 1 | 1 – Loss nicht endlich (NaN/Inf) | 885 / 16000 | bvpch43h | 20261011T160435Z-cfgd7bea1643e50-9e3bddc0 |
| NCA | circular | alien_monster+crab+lizard+microbe | 0 | 1 – Loss nicht endlich (NaN/Inf) | 540 / 16000 | fyehv0x9 | 20261011T015504Z-cfg6ee5d6e95492-41ae6ca3 |
| NCA | circular | alien_monster+crab+lizard+microbe | 1 | 3 – Loss explodiert (|loss| > 1e16) | 1249 / 16000 | 8luxmo93 | 20261009T195954Z-cfga786a346a7f9-f018ad3a |
| NCA | circular | alien_monster+crab+lizard+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 964 / 16000 | mx0fzft5 | 20261010T022215Z-cfg8e60f31c69ff-9b59f636 |
| NCA | circular | alien_monster+crab+lizard+rooster | 1 | 2 – Zustaende nicht endlich (NaN/Inf in states) | 307 / 16000 | 5tezdrf0 | 20261010T125326Z-cfg7c90913f0fa6-929edea6 |
| NCA | circular | alien_monster+crab+microbe+rooster | 0 | 1 – Loss nicht endlich (NaN/Inf) | 464 / 16000 | xtqrf4jg | 20261012T191329Z-cfgc064dbddd408-5f3a3e76 |
| NCA | circular | alien_monster+crab+microbe+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 1625 / 16000 | vgntewvw | 20261012T154023Z-cfgf56c444cdf32-5c9b4f29 |
| NCA | circular | alien_monster+lizard+microbe+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 982 / 16000 | uzomin45 | 20261011T133252Z-cfg8136ef470554-79e8ad94 |
| NCA | circular | alien_monster+lizard+microbe+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 931 / 16000 | bv0z2spi | 20261009T162842Z-cfgccafb2e4de4e-b3ace835 |
| NCA | circular | crab+lizard+microbe+rooster | 0 | 1 – Loss nicht endlich (NaN/Inf) | 3184 / 16000 | tjb7mk1i | 20261010T142806Z-cfg7159803794de-4d26e474 |
| NCA | circular | crab+lizard+microbe+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 392 / 16000 | qfxjlqrj | 20261012T133200Z-cfgc282dd8cf079-f470ba39 |
| NCA | circular | alien_monster+crab+lizard+microbe+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 4251 / 16000 | 7ddxr1i2 | 20261009T203631Z-cfgb006649590cf-df830b5c |
| NCA | circular | alien_monster+crab+lizard+microbe+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 1614 / 16000 | ytjc4d3j | 20261010T183846Z-cfg644f5b4eb71b-407ef58f |
| NCA | replicate | alien_monster+crab | 0 | 1 – Loss nicht endlich (NaN/Inf) | 6356 / 16000 | b01lxn2v | 20261011T203721Z-cfgee44c213450a-b34f0694 |
| NCA | replicate | alien_monster+crab | 1 | 1 – Loss nicht endlich (NaN/Inf) | 379 / 16000 | c57ll9jo | 20261012T165322Z-cfg20e62d437590-07c4bb5a |
| NCA | replicate | alien_monster+lizard | 0 | 2 – Zustaende nicht endlich (NaN/Inf in states) | 295 / 16000 | k3trieaa | 20261011T102351Z-cfg73ee4b65e2ea-d68e6ffd |
| NCA | replicate | alien_monster+lizard | 1 | 3 – Loss explodiert (|loss| > 1e16) | 340 / 16000 | gkxmn86n | 20261010T070432Z-cfg435878442973-11f4206b |
| NCA | replicate | alien_monster+microbe | 0 | 3 – Loss explodiert (|loss| > 1e16) | 390 / 16000 | 5kxme6sf | 20261012T131700Z-cfgec1fefb5f7bf-880ea4da |
| NCA | replicate | alien_monster+microbe | 1 | 2 – Zustaende nicht endlich (NaN/Inf in states) | 395 / 16000 | krvm4ecc | 20261011T020714Z-cfg80f999ede90f-5e4bc3d0 |
| NCA | replicate | alien_monster+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 439 / 16000 | 1m3svs5l | 20261012T121526Z-cfg77411fbc2be6-2035c1dd |
| NCA | replicate | alien_monster+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 885 / 16000 | 3du8xzg7 | 20261011T233401Z-cfgdde8df173fcf-3740bce2 |
| NCA | replicate | crab+lizard | 0 | 2 – Zustaende nicht endlich (NaN/Inf in states) | 1763 / 16000 | snhq8iac | 20261010T173326Z-cfge2276f7e8219-46aa44fc |
| NCA | replicate | crab+lizard | 1 | 3 – Loss explodiert (|loss| > 1e16) | 6348 / 16000 | j9ad6zba | 20261012T094442Z-cfg8f056bf22cf2-1cf39398 |
| NCA | replicate | crab+microbe | 0 | 3 – Loss explodiert (|loss| > 1e16) | 1126 / 16000 | hde3oiyi | 20261012T154433Z-cfg96dcc9cd3142-b56f0b46 |
| NCA | replicate | crab+microbe | 1 | 3 – Loss explodiert (|loss| > 1e16) | 973 / 16000 | hi6d4itl | 20261011T113145Z-cfgf3ac0ea5d732-02f12dc8 |
| NCA | replicate | crab+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 3280 / 16000 | mhqi4sbr | 20261010T145537Z-cfgeb33264c5a75-8e1af48f |
| NCA | replicate | crab+rooster | 1 | 1 – Loss nicht endlich (NaN/Inf) | 3297 / 16000 | pfhlp5jo | 20261011T200830Z-cfge13e9da542c5-1a04d850 |
| NCA | replicate | lizard+microbe | 0 | 3 – Loss explodiert (|loss| > 1e16) | 3555 / 16000 | 6toq1asu | 20261010T111326Z-cfgf771f1fde929-83b7e593 |
| NCA | replicate | lizard+microbe | 1 | 1 – Loss nicht endlich (NaN/Inf) | 305 / 16000 | 7xxi2riq | 20261012T214507Z-cfgc56d69b0ac60-8d90339d |
| NCA | replicate | lizard+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 632 / 16000 | zxpn3qvt | 20261011T021500Z-cfg6ba01425c2dd-27daa61f |
| NCA | replicate | lizard+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 1565 / 16000 | en4y6ag5 | 20261010T151344Z-cfg09d193cb6b15-c1be7186 |
| NCA | replicate | microbe+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 452 / 16000 | 97a4lczr | 20261010T055403Z-cfg6cb690338324-3a675364 |
| NCA | replicate | microbe+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 735 / 16000 | zgseyeaz | 20261011T214850Z-cfg4f831c93c33a-bf50d823 |
| NCA | replicate | alien_monster+crab+lizard | 0 | 3 – Loss explodiert (|loss| > 1e16) | 3678 / 16000 | szpd2ub8 | 20261010T205153Z-cfg27af830a3553-3a190862 |
| NCA | replicate | alien_monster+crab+lizard | 1 | 3 – Loss explodiert (|loss| > 1e16) | 767 / 16000 | ptashcue | 20261012T131923Z-cfg8805b3a082eb-3448a996 |
| NCA | replicate | alien_monster+crab+microbe | 0 | 3 – Loss explodiert (|loss| > 1e16) | 819 / 16000 | 0y3q0adf | 20261009T222422Z-cfgf65ae898f2b1-9b66eca3 |
| NCA | replicate | alien_monster+crab+microbe | 1 | 3 – Loss explodiert (|loss| > 1e16) | 12747 / 16000 | g478nv9m | 20261010T000141Z-cfga9a96fe2b100-7f3b010f |
| NCA | replicate | alien_monster+crab+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 965 / 16000 | msv0gb5f | 20261010T201618Z-cfgec43d4b1d709-b662a876 |
| NCA | replicate | alien_monster+crab+rooster | 1 | 1 – Loss nicht endlich (NaN/Inf) | 1769 / 16000 | yecse1yr | 20261012T162229Z-cfge0558645f36e-94805f7e |
| NCA | replicate | alien_monster+lizard+microbe | 0 | 2 – Zustaende nicht endlich (NaN/Inf in states) | 3120 / 16000 | 51jcc3p7 | 20261011T072526Z-cfgb6ee1001bfff-7cee44a3 |
| NCA | replicate | alien_monster+lizard+microbe | 1 | 3 – Loss explodiert (|loss| > 1e16) | 381 / 16000 | bcj0ywyr | 20261012T185942Z-cfg4b2a06afae79-7407f3a1 |
| NCA | replicate | alien_monster+lizard+rooster | 0 | 1 – Loss nicht endlich (NaN/Inf) | 2031 / 16000 | rt7i8g1i | 20261010T002707Z-cfga3ff7f16f806-c6243139 |
| NCA | replicate | alien_monster+lizard+rooster | 1 | 1 – Loss nicht endlich (NaN/Inf) | 3886 / 16000 | 71dlaenm | 20261009T144416Z-cfg9afbf7dae021-40788cea |
| NCA | replicate | alien_monster+lizard+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 600 / 16000 | pov453dc | 20261011T174031Z-cfg42a37e71898b-d58973eb |
| NCA | replicate | alien_monster+microbe+rooster | 0 | 1 – Loss nicht endlich (NaN/Inf) | 3647 / 16000 | 4c7u5lg4 | 20261010T075510Z-cfgd172d4542f9d-9799a652 |
| NCA | replicate | alien_monster+microbe+rooster | 1 | 1 – Loss nicht endlich (NaN/Inf) | 633 / 16000 | apw67vun | 20261010T203219Z-cfg7ed0db2466dc-b9307756 |
| NCA | replicate | crab+lizard+microbe | 0 | 3 – Loss explodiert (|loss| > 1e16) | 4062 / 16000 | 860z94ql | 20261010T114727Z-cfgf59cb3d84cd9-a5c7b28f |
| NCA | replicate | crab+lizard+microbe | 1 | 3 – Loss explodiert (|loss| > 1e16) | 3643 / 16000 | voqhr2z2 | 20261009T170634Z-cfgd5f50cb0232a-9c3f59e2 |
| NCA | replicate | crab+lizard+rooster | 0 | 2 – Zustaende nicht endlich (NaN/Inf in states) | 876 / 16000 | 6sp72gmd | 20261011T154119Z-cfgbfe7a6f00f1f-fa35d430 |
| NCA | replicate | crab+lizard+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 944 / 16000 | u8twjy39 | 20261010T054037Z-cfg0a4c39f5d4b2-f96aa0bc |
| NCA | replicate | crab+microbe+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 9775 / 16000 | sopqfjf3 | 20261011T113410Z-cfg8d7b8409000e-7f431992 |
| NCA | replicate | crab+microbe+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 881 / 16000 | jhlb05xj | 20261012T152037Z-cfge880c3536c1b-48afe42f |
| NCA | replicate | lizard+microbe+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 4380 / 16000 | yrhj1wf2 | 20261010T185831Z-cfg99079e815bc3-3123e058 |
| NCA | replicate | alien_monster+crab+lizard+microbe | 0 | 3 – Loss explodiert (|loss| > 1e16) | 4523 / 16000 | ralot16o | 20261010T072903Z-cfg5d36d44073fd-e7c842f7 |
| NCA | replicate | alien_monster+crab+lizard+microbe | 1 | 1 – Loss nicht endlich (NaN/Inf) | 1956 / 16000 | dd6yfm49 | 20261010T171302Z-cfg3f73802d6986-d2445c3e |
| NCA | replicate | alien_monster+crab+lizard+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 3309 / 16000 | e26hq3cn | 20261010T045805Z-cfgdcdfc4d26b2a-8272ec05 |
| NCA | replicate | alien_monster+crab+microbe+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 548 / 16000 | inb4w12k | 20261010T150434Z-cfge84659385b71-d51282cf |
| NCA | replicate | alien_monster+crab+microbe+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 6977 / 16000 | lrcitvn6 | 20261012T074509Z-cfg9def1644894d-e1c61bde |
| NCA | replicate | alien_monster+lizard+microbe+rooster | 0 | 1 – Loss nicht endlich (NaN/Inf) | 8517 / 16000 | c4ti89hk | 20261011T132031Z-cfge9528f4a7d3a-8ff09164 |
| NCA | replicate | crab+lizard+microbe+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 1113 / 16000 | 9xyutfwv | 20261009T142623Z-cfgdb8d69d23786-241d5dd0 |
| NCA | replicate | crab+lizard+microbe+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 3529 / 16000 | xzzib6xk | 20261010T143247Z-cfg741eb2239d94-dd038386 |
| NCA | replicate | crab+lizard+microbe+rooster | 1 | 1 – Loss nicht endlich (NaN/Inf) | 1025 / 16000 | spra7482 | 20261010T080626Z-cfg666a2df360e4-eee41977 |
| NCA | replicate | alien_monster+crab+lizard+microbe+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 420 / 16000 | 7nsu39ea | 20261012T152021Z-cfgac5f114390ac-7779a3f6 |
| NCA | replicate | alien_monster+crab+lizard+microbe+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 2856 / 16000 | lp20co7t | 20261010T153015Z-cfg27030d0aa93a-ce0ac524 |
| gNCA | circular | alien_monster+crab | 0 | 3 – Loss explodiert (|loss| > 1e16) | 1369 / 16000 | 2r4irp27 | 20261012T014805Z-cfga8ea69fd744f-f023a149 |
| gNCA | circular | crab+microbe | 0 | 3 – Loss explodiert (|loss| > 1e16) | 4546 / 16000 | 9jc8bemm | 20261012T105006Z-cfg09dfe183f5fd-5d5199e6 |
| gNCA | circular | lizard+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 6437 / 16000 | qquiipsx | 20261012T122637Z-cfgc919dbf76fb4-b687e8ed |
| gNCA | circular | microbe+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 3328 / 16000 | ceduapux | 20261011T035105Z-cfgd2a667397928-03d4073c |
| gNCA | circular | alien_monster+microbe+rooster | 1 | 1 – Loss nicht endlich (NaN/Inf) | 6161 / 16000 | 8gy55i8q | 20261010T081906Z-cfg6a641fa7726e-615dcf58 |
| gNCA | circular | crab+lizard+microbe | 0 | 1 – Loss nicht endlich (NaN/Inf) | 3029 / 16000 | 8hj0dtez | 20261012T140200Z-cfgdeb8582edfb3-0db41787 |
| gNCA | circular | lizard+microbe+rooster | 0 | 1 – Loss nicht endlich (NaN/Inf) | 7620 / 16000 | yrbygg9s | 20261012T004546Z-cfge64015226266-67eda344 |
| gNCA | circular | alien_monster+crab+lizard+microbe+rooster | 0 | 1 – Loss nicht endlich (NaN/Inf) | 3559 / 16000 | y09amwmv | 20261011T025212Z-cfge0566fa5e54a-35495c35 |
| gNCA | replicate | alien_monster+crab+lizard+microbe | 0 | 1 – Loss nicht endlich (NaN/Inf) | 2988 / 16000 | 9jlp5g5k | 20261010T072648Z-cfgb70b86828428-27314b38 |

### Funktionierende Läufe

| Familie | Padding | Attraktoren | Repeat | best loss | beste Iteration | Run-ID |
|---|---|---|---|---|---|---|
| NCA | circular | crab+rooster | 0 | 0.009272 | 1478 | 20261009T162857Z-cfgeb37d0fc7b84-86b36d0f |
| NCA | circular | alien_monster+microbe+rooster | 0 | 0.008391 | 11110 | 20261012T050741Z-cfg77dbd89c7f2a-19a9190c |
| NCA | circular | crab+lizard+microbe | 0 | 0.01375 | 430 | 20261012T044705Z-cfgb637fae3b7b3-a0c69bf2 |
| NCA | replicate | lizard+microbe+rooster | 1 | 0.01102 | 1956 | 20261012T073819Z-cfge659bce47db4-e2e049ef |
| NCA | replicate | alien_monster+crab+lizard+rooster | 0 | 0.0115 | 385 | 20261012T143319Z-cfg7d6f1d0545d4-35aee7a2 |
| NCA | replicate | alien_monster+lizard+microbe+rooster | 1 | 0.01242 | 1083 | 20261011T074622Z-cfg7b3a763ce0b7-513ec55e |
| gNCA | circular | alien_monster+crab | 1 | 0.00171 | 12135 | 20261012T132346Z-cfg31699f23d032-82037ba5 |
| gNCA | circular | alien_monster+lizard | 0 | 0.001579 | 15683 | 20261011T195149Z-cfgc8bcdca116a9-2acf227a |
| gNCA | circular | alien_monster+lizard | 1 | 0.002057 | 11917 | 20261009T190940Z-cfge8b6217c4dd8-7211121a |
| gNCA | circular | alien_monster+microbe | 0 | 0.01199 | 1143 | 20261010T123230Z-cfg983350106d21-ed008e2c |
| gNCA | circular | alien_monster+microbe | 1 | 0.002814 | 15507 | 20261011T235022Z-cfg5fdbac141635-a985066e |
| gNCA | circular | alien_monster+rooster | 0 | 0.002542 | 15949 | 20261010T035618Z-cfg99f38dc7bc27-41d2234a |
| gNCA | circular | alien_monster+rooster | 1 | 0.002859 | 14768 | 20261012T112048Z-cfg0883f89b9565-66600d37 |
| gNCA | circular | crab+lizard | 0 | 0.001352 | 14093 | 20261010T023311Z-cfgabba70415787-f2051eed |
| gNCA | circular | crab+lizard | 1 | 0.001511 | 10174 | 20261010T205028Z-cfg4180b0e9fcd7-1fa4374a |
| gNCA | circular | crab+microbe | 1 | 0.001924 | 15398 | 20261010T153533Z-cfg8f411fc11f0a-6fdff526 |
| gNCA | circular | crab+rooster | 0 | 0.00235 | 15933 | 20261012T194851Z-cfg7ca1867a552d-14d33e90 |
| gNCA | circular | crab+rooster | 1 | 0.002042 | 12646 | 20261010T053116Z-cfg07045cb48422-5d933405 |
| gNCA | circular | lizard+microbe | 0 | 0.004172 | 15221 | 20261010T071323Z-cfga8ce28ebfa90-370fd596 |
| gNCA | circular | lizard+microbe | 1 | 0.001571 | 15126 | 20261010T082221Z-cfg9ea091bd693a-3630290f |
| gNCA | circular | lizard+rooster | 1 | 0.002768 | 15614 | 20261012T031159Z-cfg2894f8bb6585-de674ed7 |
| gNCA | circular | microbe+rooster | 1 | 0.003366 | 12552 | 20261011T215601Z-cfg0fb6f17b08c8-0e1368d1 |
| gNCA | circular | alien_monster+crab+lizard | 0 | 0.008609 | 2615 | 20261011T104437Z-cfg2ba33e20ed08-887ae3b9 |
| gNCA | circular | alien_monster+crab+lizard | 1 | 0.001693 | 14821 | 20261010T172852Z-cfgbb6c228fae83-bc17aaaa |
| gNCA | circular | alien_monster+crab+microbe | 0 | 0.001818 | 15880 | 20261011T144916Z-cfg8f9f0d0d63af-28878092 |
| gNCA | circular | alien_monster+crab+microbe | 1 | 0.00695 | 15901 | 20261009T213510Z-cfg632267b4d599-aacc71bd |
| gNCA | circular | alien_monster+crab+rooster | 0 | 0.001695 | 12465 | 20261011T120011Z-cfg4c44da7387de-c52a0db3 |
| gNCA | circular | alien_monster+crab+rooster | 1 | 0.008896 | 1225 | 20261010T225208Z-cfgcd98f49e1de5-c8af22ad |
| gNCA | circular | alien_monster+lizard+microbe | 0 | 0.003066 | 15245 | 20261011T164312Z-cfg051caeca92df-cb93f12f |
| gNCA | circular | alien_monster+lizard+microbe | 1 | 0.001861 | 15903 | 20261009T190816Z-cfg3739ff01d5e2-f1df7ec0 |
| gNCA | circular | alien_monster+lizard+rooster | 0 | 0.002537 | 7985 | 20261011T053736Z-cfg0f1eb3404189-9b402dde |
| gNCA | circular | alien_monster+lizard+rooster | 1 | 0.002383 | 5554 | 20261010T160437Z-cfg96d9016d851a-2d289b5a |
| gNCA | circular | alien_monster+microbe+rooster | 0 | 0.00351 | 6862 | 20261012T022321Z-cfgf5dc6ffd8da0-25bc8f8f |
| gNCA | circular | crab+lizard+microbe | 1 | 0.001646 | 15960 | 20261011T043417Z-cfg512e912589c8-76eb12ec |
| gNCA | circular | crab+lizard+rooster | 0 | 0.002711 | 3827 | 20261012T015630Z-cfge85da669c016-c2f3611f |
| gNCA | circular | crab+lizard+rooster | 1 | 0.001714 | 9163 | 20261010T230328Z-cfg0ca3ada126ca-9ddfd387 |
| gNCA | circular | crab+microbe+rooster | 0 | 0.001735 | 14595 | 20261010T222945Z-cfgeb828d820aee-b3ed7785 |
| gNCA | circular | crab+microbe+rooster | 1 | 0.001996 | 14153 | 20261011T191026Z-cfg2e3405020dc3-fa90ebcd |
| gNCA | circular | lizard+microbe+rooster | 1 | 0.001487 | 15116 | 20261010T025047Z-cfg0987eb8cbdad-416dd1e9 |
| gNCA | circular | alien_monster+crab+lizard+microbe | 0 | 0.005338 | 14324 | 20261010T005112Z-cfg3b3732e4e03d-2da70cab |
| gNCA | circular | alien_monster+crab+lizard+microbe | 1 | 0.001931 | 14778 | 20261010T101948Z-cfga89c5f27b44c-47720a5b |
| gNCA | circular | alien_monster+crab+lizard+rooster | 0 | 0.003209 | 10136 | 20261011T082610Z-cfgeb052c6354f6-f2edc845 |
| gNCA | circular | alien_monster+crab+lizard+rooster | 1 | 0.001712 | 15803 | 20261011T175136Z-cfgb57d84132d78-6296e8e2 |
| gNCA | circular | alien_monster+crab+microbe+rooster | 0 | 0.006327 | 4762 | 20261009T195903Z-cfg108cb7336460-e233e113 |
| gNCA | circular | alien_monster+crab+microbe+rooster | 1 | 0.004121 | 15941 | 20261010T225005Z-cfg76de3724deab-58a207eb |
| gNCA | circular | alien_monster+lizard+microbe+rooster | 0 | 0.002906 | 14583 | 20261010T185341Z-cfg2b2863fb8683-e4b09bbb |
| gNCA | circular | alien_monster+lizard+microbe+rooster | 1 | 0.001772 | 14999 | 20261010T203002Z-cfg1fc37cfc8da6-6da031e5 |
| gNCA | circular | crab+lizard+microbe+rooster | 0 | 0.01268 | 243 | 20261010T174936Z-cfge68ad05fec50-6e7d88e3 |
| gNCA | circular | crab+lizard+microbe+rooster | 1 | 0.004772 | 6912 | 20261010T123500Z-cfg96360159593c-2f0d7baf |
| gNCA | circular | alien_monster+crab+lizard+microbe+rooster | 1 | 0.005384 | 6106 | 20261011T052318Z-cfg2b4c04c839f2-f69923ee |
| gNCA | replicate | alien_monster+crab | 0 | 0.001187 | 13275 | 20261012T155438Z-cfg852aaf695712-59f73314 |
| gNCA | replicate | alien_monster+crab | 1 | 0.001277 | 13553 | 20261011T082434Z-cfg4f141a87ca46-5a54cb63 |
| gNCA | replicate | alien_monster+lizard | 0 | 0.001356 | 12017 | 20261012T111936Z-cfgb0c53da796e9-a14f21fb |
| gNCA | replicate | alien_monster+lizard | 1 | 0.001183 | 15748 | 20261011T235939Z-cfgc50985d82674-84efd6fd |
| gNCA | replicate | alien_monster+microbe | 0 | 0.001434 | 15621 | 20261010T015409Z-cfge3ae5935363e-8c8f88cd |
| gNCA | replicate | alien_monster+microbe | 1 | 0.001235 | 15239 | 20261011T134042Z-cfg39519af44c62-72ee41ea |
| gNCA | replicate | alien_monster+rooster | 0 | 0.001185 | 9368 | 20261012T170018Z-cfg870b44681de6-b96926b9 |
| gNCA | replicate | alien_monster+rooster | 1 | 0.001027 | 14971 | 20261009T175802Z-cfgedfceb92c7c9-64e2e081 |
| gNCA | replicate | crab+lizard | 0 | 0.001563 | 11563 | 20261010T130307Z-cfg4680d2e15633-fde7a895 |
| gNCA | replicate | crab+lizard | 1 | 0.001203 | 15825 | 20261010T055428Z-cfgcaef9081e541-5343a49d |
| gNCA | replicate | crab+microbe | 0 | 0.00205 | 15775 | 20261010T201857Z-cfg2558b2c2ba3a-35880deb |
| gNCA | replicate | crab+microbe | 1 | 0.002022 | 15755 | 20261011T032238Z-cfg3d017c9db815-bc9cfe55 |
| gNCA | replicate | crab+rooster | 0 | 0.002247 | 15941 | 20261011T005237Z-cfgaa54356c53bd-5ba291ef |
| gNCA | replicate | crab+rooster | 1 | 0.00132 | 15992 | 20261011T012127Z-cfgaa1bd69e749a-a59978f6 |
| gNCA | replicate | lizard+microbe | 0 | 0.00143 | 13370 | 20261011T212704Z-cfgac53cc06242d-dd5c7149 |
| gNCA | replicate | lizard+microbe | 1 | 0.001629 | 15163 | 20261012T101549Z-cfg559a23578261-ea84dc4f |
| gNCA | replicate | lizard+rooster | 0 | 0.0009851 | 15291 | 20261011T224424Z-cfg59e588bb3dd2-0c676e4b |
| gNCA | replicate | lizard+rooster | 1 | 0.0009715 | 13244 | 20261009T210717Z-cfgaa35f24212e5-0782521d |
| gNCA | replicate | microbe+rooster | 0 | 0.001701 | 13503 | 20261012T150339Z-cfg0309e95d76eb-9a51a79c |
| gNCA | replicate | microbe+rooster | 1 | 0.001411 | 14970 | 20261010T091124Z-cfg3d43ae592d45-7e84ae30 |
| gNCA | replicate | alien_monster+crab+lizard | 0 | 0.001269 | 15518 | 20261012T161835Z-cfgc4398cfffbe6-7587bddc |
| gNCA | replicate | alien_monster+crab+lizard | 1 | 0.001221 | 15936 | 20261011T161900Z-cfge805e2c302d0-79e55f89 |
| gNCA | replicate | alien_monster+crab+microbe | 0 | 0.002274 | 14578 | 20261011T134548Z-cfg8c81ee143c78-bce3027c |
| gNCA | replicate | alien_monster+crab+microbe | 1 | 0.001877 | 14700 | 20261010T092910Z-cfg90c2708f111e-e4c3c5d7 |
| gNCA | replicate | alien_monster+crab+rooster | 0 | 0.002111 | 15745 | 20261011T194939Z-cfg007737a325dc-618ffa3f |
| gNCA | replicate | alien_monster+crab+rooster | 1 | 0.001265 | 14168 | 20261010T085727Z-cfg18d2e7f93c86-59fcd4dc |
| gNCA | replicate | alien_monster+lizard+microbe | 0 | 0.001782 | 14249 | 20261009T225759Z-cfgaba9e99b8fb0-faf0070d |
| gNCA | replicate | alien_monster+lizard+microbe | 1 | 0.005748 | 15485 | 20261011T053926Z-cfged37287e20a5-31e028c4 |
| gNCA | replicate | alien_monster+lizard+rooster | 0 | 0.001188 | 15792 | 20261010T154711Z-cfg98c2d651c7d7-c4565da5 |
| gNCA | replicate | alien_monster+lizard+rooster | 1 | 0.001449 | 15985 | 20261010T043403Z-cfg8f307f06293b-92c3fc6d |
| gNCA | replicate | alien_monster+microbe+rooster | 0 | 0.001594 | 15486 | 20261011T014053Z-cfga613adfe1a72-8b723426 |
| gNCA | replicate | alien_monster+microbe+rooster | 1 | 0.001597 | 13608 | 20261012T044545Z-cfg0dc80f33c049-f331efea |
| gNCA | replicate | crab+lizard+microbe | 0 | 0.001548 | 15464 | 20261010T114028Z-cfg86ed5486cb23-11a718d8 |
| gNCA | replicate | crab+lizard+microbe | 1 | 0.001566 | 14874 | 20261011T102431Z-cfgef0a37f3fa16-180c7dc2 |
| gNCA | replicate | crab+lizard+rooster | 0 | 0.001382 | 14673 | 20261012T113959Z-cfg503766e44a26-a573e7f9 |
| gNCA | replicate | crab+lizard+rooster | 1 | 0.002031 | 8590 | 20261012T170023Z-cfgc239e08200cd-fb161fc0 |
| gNCA | replicate | crab+microbe+rooster | 0 | 0.001612 | 15605 | 20261011T083736Z-cfg6427857285b0-3cbe5231 |
| gNCA | replicate | crab+microbe+rooster | 1 | 0.001589 | 13089 | 20261011T151407Z-cfg0f4620658f89-27773bf1 |
| gNCA | replicate | lizard+microbe+rooster | 0 | 0.001761 | 13792 | 20261009T230734Z-cfg777ba3ba0ce7-2444a2b1 |
| gNCA | replicate | lizard+microbe+rooster | 1 | 0.001412 | 15372 | 20261012T054213Z-cfg9abc9e982ed9-4752c085 |
| gNCA | replicate | alien_monster+crab+lizard+microbe | 1 | 0.002268 | 15470 | 20261011T093411Z-cfgbaf4b72d778f-86bf8e70 |
| gNCA | replicate | alien_monster+crab+lizard+rooster | 0 | 0.001675 | 15950 | 20261011T033814Z-cfg8a5d220069ca-085b8b3a |
| gNCA | replicate | alien_monster+crab+lizard+rooster | 1 | 0.00159 | 15461 | 20261010T150935Z-cfg8a00de9e47e2-1aa15e1e |
| gNCA | replicate | alien_monster+crab+microbe+rooster | 0 | 0.001984 | 14664 | 20261011T215907Z-cfg2685f57dcda2-d045c357 |
| gNCA | replicate | alien_monster+crab+microbe+rooster | 1 | 0.002111 | 15864 | 20261012T074110Z-cfg4348306cc9bc-623af410 |
| gNCA | replicate | alien_monster+lizard+microbe+rooster | 0 | 0.00144 | 15543 | 20261012T131840Z-cfgc1394903f46f-f568c298 |
| gNCA | replicate | alien_monster+lizard+microbe+rooster | 1 | 0.002366 | 15903 | 20261011T180727Z-cfgba2867d43e31-be7c5b2c |
| gNCA | replicate | crab+lizard+microbe+rooster | 0 | 0.002969 | 15988 | 20261011T235302Z-cfg98afbd3f7ba2-b4ba2237 |
| gNCA | replicate | crab+lizard+microbe+rooster | 1 | 0.001703 | 14995 | 20261009T215701Z-cfg7e6841e86ed7-643c627a |
| gNCA | replicate | alien_monster+crab+lizard+microbe+rooster | 0 | 0.002219 | 15694 | 20261012T020020Z-cfg49b78daeb330-a22d9428 |
| gNCA | replicate | alien_monster+crab+lizard+microbe+rooster | 1 | 0.002312 | 15572 | 20261012T085203Z-cfgd58d284371ad-415f8dd2 |

## v2 (206 Läufe: 99 funktioniert, 107 abgebrochen)

| Familie | Padding | funktioniert | abgebrochen | davon Err1 | Err2 | Err3 |
|---|---|---|---|---|---|---|
| NCA | circular | 1 | 51 | 15 | 3 | 33 |
| NCA | replicate | 9 | 43 | 11 | 0 | 32 |
| gNCA | circular | 44 | 7 | 1 | 0 | 6 |
| gNCA | replicate | 45 | 6 | 0 | 0 | 6 |

### Abgebrochene Läufe

| Familie | Padding | Attraktoren | Repeat | Fehler | letzter OK-Checkpoint | W&B | Run-ID |
|---|---|---|---|---|---|---|---|
| NCA | circular | alien_monster+crab | 1 | 1 – Loss nicht endlich (NaN/Inf) | 7764 / 16000 | tf6mzx8h | 20261017T075326Z-cfg2c655411ca00-4e7fead7 |
| NCA | circular | alien_monster+lizard | 0 | 1 – Loss nicht endlich (NaN/Inf) | 573 / 16000 | 2u3wf0c9 | 20261014T172549Z-cfgb3c0320b00e9-f34f3484 |
| NCA | circular | alien_monster+lizard | 1 | 3 – Loss explodiert (|loss| > 1e16) | 871 / 16000 | f32dp8ja | 20261015T092316Z-cfg08250f555b2b-47af1208 |
| NCA | circular | alien_monster+microbe | 0 | 1 – Loss nicht endlich (NaN/Inf) | 1466 / 16000 | 0nppx76c | 20261017T053658Z-cfgc6417ee4e698-ccec3269 |
| NCA | circular | alien_monster+microbe | 1 | 1 – Loss nicht endlich (NaN/Inf) | 359 / 16000 | 67u8lvob | 20261016T101516Z-cfg4d016d34e622-7d88bf1b |
| NCA | circular | alien_monster+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 1541 / 16000 | a2jhlv7j | 20261015T090504Z-cfge9d3d1955b57-488a0736 |
| NCA | circular | alien_monster+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 946 / 16000 | q3xd2irl | 20261014T200046Z-cfg6a568999601e-1e5cc5b7 |
| NCA | circular | crab+lizard | 0 | 3 – Loss explodiert (|loss| > 1e16) | 2542 / 16000 | wwq8ovrj | 20261014T103526Z-cfg1d85a177f86d-f0cedc2d |
| NCA | circular | crab+lizard | 1 | 3 – Loss explodiert (|loss| > 1e16) | 8694 / 16000 | ww6s7mj4 | 20261015T085058Z-cfg3c7bdb28de5e-06b96ffb |
| NCA | circular | crab+microbe | 0 | 1 – Loss nicht endlich (NaN/Inf) | 1587 / 16000 | mvxbi52b | 20261016T073836Z-cfgf9e778bbb60d-f506f91f |
| NCA | circular | crab+microbe | 1 | 3 – Loss explodiert (|loss| > 1e16) | 905 / 16000 | rw3q8d55 | 20261014T215735Z-cfg5b276142e517-4b6c7a5e |
| NCA | circular | crab+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 3007 / 16000 | v9v7u5ab | 20261014T055455Z-cfg4a08da76318d-0ade5eba |
| NCA | circular | crab+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 1610 / 16000 | m62sq6yd | 20261017T164605Z-cfg9f499ef7f0e2-9af10d11 |
| NCA | circular | lizard+microbe | 0 | 3 – Loss explodiert (|loss| > 1e16) | 2499 / 16000 | 585onvsz | 20261014T220906Z-cfg4657d86b696c-ba07ca23 |
| NCA | circular | lizard+microbe | 1 | 3 – Loss explodiert (|loss| > 1e16) | 1672 / 16000 | jfs8qpsf | 20261014T000319Z-cfgd7d5fa140b12-bb5bfe16 |
| NCA | circular | lizard+rooster | 0 | 1 – Loss nicht endlich (NaN/Inf) | 2405 / 16000 | rgjc895e | 20261015T063824Z-cfg6a713c8c062e-9e0c78be |
| NCA | circular | lizard+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 1636 / 16000 | 27jc83ne | 20261016T163025Z-cfgc00e0bf1fb24-2da8b058 |
| NCA | circular | microbe+rooster | 0 | 1 – Loss nicht endlich (NaN/Inf) | 5087 / 16000 | e282g9gc | 20261014T161850Z-cfg8ec2769ba5f0-31a4b815 |
| NCA | circular | microbe+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 5569 / 16000 | 6y44v8pw | 20261014T220846Z-cfg47656cb4ca34-f15a1acd |
| NCA | circular | alien_monster+crab+lizard | 0 | 3 – Loss explodiert (|loss| > 1e16) | 3711 / 16000 | byamwvm8 | 20261016T035929Z-cfg35f284470a43-1459fb23 |
| NCA | circular | alien_monster+crab+lizard | 1 | 3 – Loss explodiert (|loss| > 1e16) | 3461 / 16000 | z4dy705u | 20261015T032250Z-cfg87f3454d8263-b4eea6f5 |
| NCA | circular | alien_monster+crab+microbe | 0 | 1 – Loss nicht endlich (NaN/Inf) | 2302 / 16000 | os50wt5r | 20261015T152301Z-cfgf8983fbd900b-ab854c3e |
| NCA | circular | alien_monster+crab+microbe | 1 | 3 – Loss explodiert (|loss| > 1e16) | 11435 / 16000 | zbzm2vwo | 20261014T010915Z-cfgc9968abb7033-bc86114d |
| NCA | circular | alien_monster+crab+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 9922 / 16000 | jf70rkhv | 20261017T035233Z-cfgb99283f5ddd1-f09f1708 |
| NCA | circular | alien_monster+crab+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 3414 / 16000 | u3w0esdb | 20261017T055718Z-cfge4da56294be9-1fa5a076 |
| NCA | circular | alien_monster+lizard+microbe | 0 | 3 – Loss explodiert (|loss| > 1e16) | 2371 / 16000 | cdkratsi | 20261017T022523Z-cfgacc753485952-dee25091 |
| NCA | circular | alien_monster+lizard+microbe | 1 | 2 – Zustaende nicht endlich (NaN/Inf in states) | 5678 / 16000 | zcalzx9p | 20261016T070504Z-cfgb19b96e74ba8-4ce58f8b |
| NCA | circular | alien_monster+lizard+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 6106 / 16000 | up6hk44w | 20261017T061803Z-cfg8286dab69a6a-937c1533 |
| NCA | circular | alien_monster+lizard+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 1721 / 16000 | xold5rnw | 20261016T034143Z-cfg61b4f82eb42a-37122b56 |
| NCA | circular | alien_monster+microbe+rooster | 0 | 1 – Loss nicht endlich (NaN/Inf) | 499 / 16000 | duno957b | 20261016T191424Z-cfgd1b6c692134d-34fc192d |
| NCA | circular | alien_monster+microbe+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 1573 / 16000 | pjw5sgbx | 20261016T143633Z-cfg038317a474a3-801bc4ac |
| NCA | circular | crab+lizard+microbe | 0 | 1 – Loss nicht endlich (NaN/Inf) | 3216 / 16000 | elm6apdb | 20261014T164317Z-cfge6efa9489136-7f4d9756 |
| NCA | circular | crab+lizard+microbe | 1 | 1 – Loss nicht endlich (NaN/Inf) | 4197 / 16000 | sdtyzye5 | 20261014T161000Z-cfgfbe076ff10f8-89204283 |
| NCA | circular | crab+lizard+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 6437 / 16000 | 144jh42b | 20261017T103245Z-cfg2bfc0a2b8afa-24c2a578 |
| NCA | circular | crab+lizard+rooster | 1 | 1 – Loss nicht endlich (NaN/Inf) | 1735 / 16000 | 8kesv6fo | 20261014T015542Z-cfg1589582fe84e-798f2c64 |
| NCA | circular | crab+microbe+rooster | 0 | 1 – Loss nicht endlich (NaN/Inf) | 3598 / 16000 | r685hf9n | 20261015T211016Z-cfgccfaa96c6951-aa216a8f |
| NCA | circular | crab+microbe+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 862 / 16000 | xfc8uzpn | 20261015T090319Z-cfgdf962e31f114-c383f8ee |
| NCA | circular | lizard+microbe+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 917 / 16000 | c0uyd1wm | 20261016T100147Z-cfg8e871a40c1f4-5dc68024 |
| NCA | circular | lizard+microbe+rooster | 1 | 1 – Loss nicht endlich (NaN/Inf) | 1027 / 16000 | mxl7tv5v | 20261017T170123Z-cfgb658232e736b-8448c4bb |
| NCA | circular | alien_monster+crab+lizard+microbe | 0 | 3 – Loss explodiert (|loss| > 1e16) | 3400 / 16000 | r7iff8fi | 20261017T145158Z-cfgf3113601e217-6af07558 |
| NCA | circular | alien_monster+crab+lizard+microbe | 1 | 3 – Loss explodiert (|loss| > 1e16) | 882 / 16000 | sc75tft2 | 20261016T065644Z-cfgcb35c8b47bdf-2937ffa8 |
| NCA | circular | alien_monster+crab+lizard+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 3878 / 16000 | wl42s320 | 20261014T140433Z-cfgd3246c71ed24-1af49e57 |
| NCA | circular | alien_monster+crab+lizard+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 2861 / 16000 | p3mwanzm | 20261015T184848Z-cfg43b1f39a2a79-7fcf33ca |
| NCA | circular | alien_monster+crab+microbe+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 3938 / 16000 | r3mbpwgx | 20261013T194922Z-cfgf89912197b90-8c54c983 |
| NCA | circular | alien_monster+crab+microbe+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 2139 / 16000 | mtn9iqwa | 20261014T031137Z-cfg09f9e73611df-67ea6d6c |
| NCA | circular | alien_monster+lizard+microbe+rooster | 0 | 2 – Zustaende nicht endlich (NaN/Inf in states) | 1832 / 16000 | xtn2yovo | 20261016T102453Z-cfg885cbc834794-f81277b1 |
| NCA | circular | alien_monster+lizard+microbe+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 4026 / 16000 | q3zkpvsi | 20261017T055513Z-cfg147ff99f97eb-a17ac58e |
| NCA | circular | crab+lizard+microbe+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 1490 / 16000 | 2r6isuqu | 20261017T055747Z-cfgb3da9eccab21-4d8fc780 |
| NCA | circular | crab+lizard+microbe+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 2824 / 16000 | tjhivqmi | 20261013T195222Z-cfg6bec249abd06-01e787df |
| NCA | circular | alien_monster+crab+lizard+microbe+rooster | 0 | 1 – Loss nicht endlich (NaN/Inf) | 5052 / 16000 | n6sxs6bi | 20261016T072436Z-cfgfda5dbe58d55-ce5567fb |
| NCA | circular | alien_monster+crab+lizard+microbe+rooster | 1 | 2 – Zustaende nicht endlich (NaN/Inf in states) | 6407 / 16000 | qmqrufri | 20261017T063826Z-cfg93236df577cc-1107cce3 |
| NCA | replicate | alien_monster+crab | 0 | 3 – Loss explodiert (|loss| > 1e16) | 805 / 16000 | 9m7qsaei | 20261018T055757Z-cfg35a732ca9841-58193e20 |
| NCA | replicate | alien_monster+crab | 1 | 3 – Loss explodiert (|loss| > 1e16) | 1434 / 16000 | diuuc76l | 20261015T032646Z-cfgf2cb93f98f76-2ceb01cc |
| NCA | replicate | alien_monster+lizard | 0 | 1 – Loss nicht endlich (NaN/Inf) | 3663 / 16000 | ibl3l8dw | 20261016T153038Z-cfga88e6e3e1309-1a2f0cff |
| NCA | replicate | alien_monster+rooster | 0 | 1 – Loss nicht endlich (NaN/Inf) | 7371 / 16000 | lt1f30xm | 20261014T064213Z-cfg5bb8dce80b9f-570c3ee2 |
| NCA | replicate | alien_monster+rooster | 1 | 1 – Loss nicht endlich (NaN/Inf) | 1703 / 16000 | yjew7ons | 20261017T235717Z-cfg33dfa2aef8fa-27d1da6f |
| NCA | replicate | crab+lizard | 0 | 3 – Loss explodiert (|loss| > 1e16) | 1132 / 16000 | jtkio43k | 20261016T121408Z-cfg3caf589d0416-230c5258 |
| NCA | replicate | crab+lizard | 1 | 3 – Loss explodiert (|loss| > 1e16) | 657 / 16000 | j0u2tzbm | 20261015T220328Z-cfge1e398445625-11252f92 |
| NCA | replicate | crab+microbe | 0 | 3 – Loss explodiert (|loss| > 1e16) | 4395 / 16000 | i6o1127r | 20261014T225844Z-cfg9d734d1d3920-0069b30e |
| NCA | replicate | crab+microbe | 1 | 3 – Loss explodiert (|loss| > 1e16) | 3491 / 16000 | 8mrvr2q2 | 20261016T150130Z-cfg59197146053e-227a74b4 |
| NCA | replicate | crab+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 8036 / 16000 | ij5n22vk | 20261013T232739Z-cfgc290e28509d8-0408a80c |
| NCA | replicate | crab+rooster | 1 | 1 – Loss nicht endlich (NaN/Inf) | 1302 / 16000 | c7igqa0x | 20261013T235005Z-cfg5d892640fc75-2f28b30e |
| NCA | replicate | lizard+microbe | 0 | 3 – Loss explodiert (|loss| > 1e16) | 15416 / 16000 | 96h1k1cx | 20261014T195529Z-cfg418aeb5f1b5a-28524eb5 |
| NCA | replicate | lizard+microbe | 1 | 3 – Loss explodiert (|loss| > 1e16) | 1611 / 16000 | rs4t0cwn | 20261014T135023Z-cfg5fffd4e50a8d-e27954b1 |
| NCA | replicate | lizard+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 5047 / 16000 | rxh4viq3 | 20261017T182253Z-cfg3acd065acf60-22bcc17b |
| NCA | replicate | lizard+rooster | 1 | 1 – Loss nicht endlich (NaN/Inf) | 5073 / 16000 | rq992zyp | 20261016T110109Z-cfgad48508505df-3aab2efa |
| NCA | replicate | microbe+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 4242 / 16000 | pbg32p29 | 20261014T161010Z-cfg60dfc64c9bfd-ebf92b9e |
| NCA | replicate | microbe+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 9634 / 16000 | uc6iszd4 | 20261015T115444Z-cfg91e107383f7e-6bd91cb4 |
| NCA | replicate | alien_monster+crab+lizard | 0 | 1 – Loss nicht endlich (NaN/Inf) | 721 / 16000 | 2a631nnc | 20261016T064524Z-cfg712a0ca20bf2-6e59ae8f |
| NCA | replicate | alien_monster+crab+lizard | 1 | 3 – Loss explodiert (|loss| > 1e16) | 3430 / 16000 | t61a7hoa | 20261016T143506Z-cfg117831d51336-dd6162fb |
| NCA | replicate | alien_monster+crab+microbe | 0 | 3 – Loss explodiert (|loss| > 1e16) | 1132 / 16000 | gssobht5 | 20261015T030650Z-cfgc211ae7f44c2-e7c7da59 |
| NCA | replicate | alien_monster+crab+microbe | 1 | 3 – Loss explodiert (|loss| > 1e16) | 5318 / 16000 | 133mvnjz | 20261015T093721Z-cfgd2f76775eecc-88957a15 |
| NCA | replicate | alien_monster+crab+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 3192 / 16000 | rqvi6ow8 | 20261015T152626Z-cfgdb02d2a524bb-9fe4435c |
| NCA | replicate | alien_monster+crab+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 1758 / 16000 | u6sbtquu | 20261014T214439Z-cfgb13a628ecb36-3faf619e |
| NCA | replicate | alien_monster+lizard+microbe | 0 | 3 – Loss explodiert (|loss| > 1e16) | 2960 / 16000 | 2y37jl0a | 20261017T063129Z-cfgfa1f0b68ec32-311433a7 |
| NCA | replicate | alien_monster+lizard+microbe | 1 | 3 – Loss explodiert (|loss| > 1e16) | 3870 / 16000 | hmgs2tfw | 20261014T030737Z-cfga6e96a04d58d-cfbad713 |
| NCA | replicate | alien_monster+lizard+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 6498 / 16000 | dsrsgklj | 20261015T092236Z-cfgba6e74f3c35e-528e7b76 |
| NCA | replicate | alien_monster+microbe+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 1189 / 16000 | wxgchcs8 | 20261015T150955Z-cfge93311727938-6387e132 |
| NCA | replicate | crab+lizard+microbe | 0 | 3 – Loss explodiert (|loss| > 1e16) | 3903 / 16000 | c8unmwe4 | 20261014T233737Z-cfgc344d665c357-3d7b3971 |
| NCA | replicate | crab+lizard+microbe | 1 | 3 – Loss explodiert (|loss| > 1e16) | 744 / 16000 | rpqb7lr3 | 20261016T112505Z-cfg1b009e83af66-45405634 |
| NCA | replicate | crab+lizard+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 1513 / 16000 | t8tqccu2 | 20261017T190306Z-cfgc546aec39e25-54f99bcb |
| NCA | replicate | crab+microbe+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 1168 / 16000 | zgplsrqz | 20261015T150616Z-cfg4acf1ff7e34b-91938a11 |
| NCA | replicate | crab+microbe+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 13930 / 16000 | 1m3ian1x | 20261014T062452Z-cfg3aa59d93ae7b-49ca4d92 |
| NCA | replicate | alien_monster+crab+lizard+microbe | 0 | 3 – Loss explodiert (|loss| > 1e16) | 14484 / 16000 | w1zknbm5 | 20261015T104545Z-cfge7b8edd8691c-3c8be16b |
| NCA | replicate | alien_monster+crab+lizard+microbe | 1 | 3 – Loss explodiert (|loss| > 1e16) | 3190 / 16000 | tyh3i633 | 20261017T030040Z-cfg066f1355a0d6-f5e07d89 |
| NCA | replicate | alien_monster+crab+lizard+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 10896 / 16000 | zhe875gs | 20261015T143544Z-cfg83a0f6995573-19eba3f3 |
| NCA | replicate | alien_monster+crab+lizard+rooster | 1 | 1 – Loss nicht endlich (NaN/Inf) | 2237 / 16000 | 830woz5w | 20261014T202910Z-cfg9180921e4a84-29d62618 |
| NCA | replicate | alien_monster+crab+microbe+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 2167 / 16000 | 398mg7aq | 20261016T114951Z-cfgb13ac677b433-9233e739 |
| NCA | replicate | alien_monster+lizard+microbe+rooster | 0 | 1 – Loss nicht endlich (NaN/Inf) | 2662 / 16000 | hkvpuo1h | 20261018T045544Z-cfg873dface0b9a-5c7bec3d |
| NCA | replicate | alien_monster+lizard+microbe+rooster | 1 | 1 – Loss nicht endlich (NaN/Inf) | 2425 / 16000 | 8istju6l | 20261013T225943Z-cfg8d66c665383c-fda14373 |
| NCA | replicate | crab+lizard+microbe+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 9669 / 16000 | yvdlylsv | 20261016T122939Z-cfg6bd301c90abb-a9d769cc |
| NCA | replicate | crab+lizard+microbe+rooster | 1 | 1 – Loss nicht endlich (NaN/Inf) | 12410 / 16000 | 6wzh0s87 | 20261015T013825Z-cfgabfd6ca5e061-8d418688 |
| NCA | replicate | alien_monster+crab+lizard+microbe+rooster | 0 | 1 – Loss nicht endlich (NaN/Inf) | 3012 / 16000 | 43p9k0eo | 20261015T015701Z-cfg504343bb21bb-2bdabf92 |
| NCA | replicate | alien_monster+crab+lizard+microbe+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 6478 / 16000 | nuedetkn | 20261018T111046Z-cfgc316c900c0f9-a8d8209d |
| gNCA | circular | alien_monster+microbe | 0 | 3 – Loss explodiert (|loss| > 1e16) | 657 / 16000 | 9mexzgny | 20261015T031515Z-cfgd4bfffd1aaf1-5818248e |
| gNCA | circular | microbe+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 12590 / 16000 | 6c95srh2 | 20261014T201524Z-cfgfed96aadfee0-6ae152ca |
| gNCA | circular | microbe+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 5139 / 16000 | 4uqg1j2f | 20261015T021423Z-cfgcb735c68bb73-1bbd0253 |
| gNCA | circular | alien_monster+crab+lizard | 0 | 3 – Loss explodiert (|loss| > 1e16) | 5820 / 16000 | 09rbzyrx | 20261016T193450Z-cfg932e5b832865-e49196f6 |
| gNCA | circular | alien_monster+lizard+microbe | 0 | 1 – Loss nicht endlich (NaN/Inf) | 795 / 16000 | vlte55yu | 20261014T065340Z-cfg882d65d9df8a-9d27e5cb |
| gNCA | circular | lizard+microbe+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 4848 / 16000 | t656vldq | 20261016T160009Z-cfg6c719b6f27ac-ae9d78e8 |
| gNCA | circular | alien_monster+crab+lizard+microbe+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 11098 / 16000 | a0wplhmx | 20261014T105007Z-cfga0c2ba021e27-ba16eea4 |
| gNCA | replicate | alien_monster+crab | 0 | 3 – Loss explodiert (|loss| > 1e16) | 9640 / 16000 | regbs03e | 20261013T213618Z-cfg40b07837609a-f37d72fc |
| gNCA | replicate | crab+microbe | 1 | 3 – Loss explodiert (|loss| > 1e16) | 10235 / 16000 | lt12cnt4 | 20261014T023219Z-cfg9311ecba25a0-a24fd71b |
| gNCA | replicate | alien_monster+crab+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 2304 / 16000 | 94lcl865 | 20261016T114006Z-cfgbcb40cab5672-93ba8ca8 |
| gNCA | replicate | alien_monster+lizard+microbe | 0 | 3 – Loss explodiert (|loss| > 1e16) | 13531 / 16000 | xgjmwyc9 | 20261014T171642Z-cfgfe4d54cd2a1d-80a14cbe |
| gNCA | replicate | alien_monster+lizard+rooster | 1 | 3 – Loss explodiert (|loss| > 1e16) | 9896 / 16000 | jbtgtl2z | 20261016T123240Z-cfge1bd369e8cd4-605cea6c |
| gNCA | replicate | alien_monster+crab+lizard+microbe+rooster | 0 | 3 – Loss explodiert (|loss| > 1e16) | 15006 / 16000 | sug98utl | 20261014T082514Z-cfgf1b95f655b41-456aac23 |

### Funktionierende Läufe

| Familie | Padding | Attraktoren | Repeat | best loss | beste Iteration | Run-ID |
|---|---|---|---|---|---|---|
| NCA | circular | alien_monster+crab | 0 | 0.01095 | 2238 | 20261014T062737Z-cfge97372baaf2a-95ccc5a1 |
| NCA | replicate | alien_monster+lizard | 1 | 0.003937 | 15973 | 20261017T111708Z-cfgd561636e8941-585baec7 |
| NCA | replicate | alien_monster+microbe | 0 | 0.003488 | 11327 | 20261015T070434Z-cfgfe0cf34a6eb6-9ca09cc6 |
| NCA | replicate | alien_monster+microbe | 1 | 0.007118 | 11735 | 20261017T003427Z-cfge9c47fd02646-f552a293 |
| NCA | replicate | alien_monster+lizard+rooster | 0 | 0.01068 | 1328 | 20261017T032156Z-cfg5460431b55e4-ffcceb32 |
| NCA | replicate | alien_monster+microbe+rooster | 0 | 0.01398 | 15979 | 20261017T121351Z-cfga407fbcbc4bb-7bc9dab2 |
| NCA | replicate | crab+lizard+rooster | 0 | 0.002787 | 15623 | 20261016T090727Z-cfgdf62571f52e8-a0f5060c |
| NCA | replicate | lizard+microbe+rooster | 0 | 0.004864 | 13346 | 20261016T005944Z-cfg230a3e8d7934-ecce7d79 |
| NCA | replicate | lizard+microbe+rooster | 1 | 0.01177 | 15790 | 20261014T074253Z-cfgb078b978a673-266598b5 |
| NCA | replicate | alien_monster+crab+microbe+rooster | 1 | 0.008342 | 15946 | 20261015T081617Z-cfg3768322ee41c-4b701014 |
| gNCA | circular | alien_monster+crab | 0 | 0.00161 | 15958 | 20261016T194535Z-cfg1779e083e849-caefa65d |
| gNCA | circular | alien_monster+crab | 1 | 0.001591 | 15340 | 20261015T123617Z-cfg0e2f9cc7c470-ffb6e695 |
| gNCA | circular | alien_monster+lizard | 0 | 0.001682 | 12903 | 20261016T151426Z-cfg0fba8a8ee6a7-79d8be47 |
| gNCA | circular | alien_monster+lizard | 1 | 0.002049 | 10192 | 20261017T204646Z-cfg3a185e47badf-f6c47d60 |
| gNCA | circular | alien_monster+microbe | 1 | 0.004933 | 15920 | 20261016T162715Z-cfg8ba55146e923-07326289 |
| gNCA | circular | alien_monster+rooster | 0 | 0.002181 | 15709 | 20261014T010833Z-cfg9cef4c9193cf-60645c21 |
| gNCA | circular | alien_monster+rooster | 1 | 0.001395 | 15590 | 20261017T040023Z-cfg4cd0c577cf62-cedb9255 |
| gNCA | circular | crab+lizard | 0 | 0.00284 | 6244 | 20261014T042747Z-cfg90c11b21a842-dbc9bfa9 |
| gNCA | circular | crab+lizard | 1 | 0.002232 | 15634 | 20261016T015132Z-cfgeb78c0ac6225-f13fee34 |
| gNCA | circular | crab+microbe | 0 | 0.002339 | 15003 | 20261018T091338Z-cfg8041f331ff6e-0f9caa36 |
| gNCA | circular | crab+microbe | 1 | 0.002288 | 9893 | 20261015T040803Z-cfg9bc66d0d5bb4-4c391d3b |
| gNCA | circular | crab+rooster | 0 | 0.001537 | 14794 | 20261017T152559Z-cfga1e7aa7143bd-42e0a776 |
| gNCA | circular | crab+rooster | 1 | 0.002124 | 15515 | 20261014T180146Z-cfgb34e87f548b4-c9baab65 |
| gNCA | circular | lizard+microbe | 0 | 0.002462 | 9758 | 20261015T095547Z-cfg57ddeec049d8-da790f37 |
| gNCA | circular | lizard+microbe | 1 | 0.002428 | 15606 | 20261016T171437Z-cfg6a4093e7e7de-ad4e4564 |
| gNCA | circular | lizard+rooster | 0 | 0.001583 | 14781 | 20261014T212053Z-cfg245bde613e9c-835207b8 |
| gNCA | circular | lizard+rooster | 1 | 0.003389 | 9501 | 20261015T031611Z-cfg05048e207964-639ec859 |
| gNCA | circular | alien_monster+crab+lizard | 1 | 0.002433 | 15991 | 20261016T165421Z-cfgd19452bed370-e3588137 |
| gNCA | circular | alien_monster+crab+microbe | 0 | 0.001956 | 15986 | 20261013T221446Z-cfg31f85826b7fe-0d9f1700 |
| gNCA | circular | alien_monster+crab+microbe | 1 | 0.001694 | 15746 | 20261014T035204Z-cfgb75f8049b94c-7e30c122 |
| gNCA | circular | alien_monster+crab+rooster | 0 | 0.001987 | 12460 | 20261016T114221Z-cfgfa418899ee79-017c4ce8 |
| gNCA | circular | alien_monster+crab+rooster | 1 | 0.001506 | 15069 | 20261016T122354Z-cfgbaf2d06e4c0d-39e36ab6 |
| gNCA | circular | alien_monster+lizard+microbe | 1 | 0.008536 | 2515 | 20261015T222810Z-cfg16833b430707-1dc276c8 |
| gNCA | circular | alien_monster+lizard+rooster | 0 | 0.003183 | 13405 | 20261015T002334Z-cfgf387db87f100-4be4a8c5 |
| gNCA | circular | alien_monster+lizard+rooster | 1 | 0.002061 | 13048 | 20261018T003610Z-cfg39e3ee37d987-ce4a9f23 |
| gNCA | circular | alien_monster+microbe+rooster | 0 | 0.002427 | 15762 | 20261017T152800Z-cfga46ac68bef10-76edcacb |
| gNCA | circular | alien_monster+microbe+rooster | 1 | 0.001746 | 14916 | 20261014T232135Z-cfgfeb4b766ab1d-59a60796 |
| gNCA | circular | crab+lizard+microbe | 0 | 0.00176 | 13965 | 20261014T040119Z-cfgb313ba6b1a6c-8261c769 |
| gNCA | circular | crab+lizard+microbe | 1 | 0.002554 | 7465 | 20261016T011810Z-cfga622048828d8-cefccb5e |
| gNCA | circular | crab+lizard+rooster | 0 | 0.002164 | 10028 | 20261015T155858Z-cfga77be8a04b27-bd918b01 |
| gNCA | circular | crab+lizard+rooster | 1 | 0.001495 | 14609 | 20261016T214337Z-cfgf8a9e8b027fa-ca592411 |
| gNCA | circular | crab+microbe+rooster | 0 | 0.004243 | 15916 | 20261014T171253Z-cfge34ee655018a-93635dff |
| gNCA | circular | lizard+microbe+rooster | 1 | 0.001549 | 15259 | 20261014T105617Z-cfg4f49bb5b009e-b08311c9 |
| gNCA | circular | alien_monster+crab+lizard+microbe | 0 | 0.002337 | 15993 | 20261015T230117Z-cfg630e5b71ecff-0583a02b |
| gNCA | circular | alien_monster+crab+lizard+microbe | 1 | 0.002268 | 15700 | 20261014T081949Z-cfg4ddc6d3fd8ae-dc14e617 |
| gNCA | circular | alien_monster+crab+lizard+rooster | 0 | 0.001708 | 14294 | 20261017T072019Z-cfg2a7df5d2e13a-a93d4a12 |
| gNCA | circular | alien_monster+crab+lizard+rooster | 1 | 0.001418 | 15864 | 20261016T232128Z-cfg398499fbbd60-3f267e1e |
| gNCA | circular | alien_monster+crab+microbe+rooster | 0 | 0.001772 | 15957 | 20261015T185123Z-cfgbd9a62878790-bfc0163e |
| gNCA | circular | alien_monster+crab+microbe+rooster | 1 | 0.002331 | 14544 | 20261017T152610Z-cfg8f4234df13b1-47948aae |
| gNCA | circular | alien_monster+lizard+microbe+rooster | 0 | 0.001579 | 15907 | 20261014T005613Z-cfg7704c3a63fb9-4533847f |
| gNCA | circular | alien_monster+lizard+microbe+rooster | 1 | 0.002903 | 15934 | 20261016T175804Z-cfga243567b66b9-ec8b15df |
| gNCA | circular | crab+lizard+microbe+rooster | 0 | 0.002512 | 12579 | 20261015T180811Z-cfg8070c2a3e641-878ce385 |
| gNCA | circular | crab+lizard+microbe+rooster | 1 | 0.001819 | 15788 | 20261016T040911Z-cfg1877b1beffda-596e589e |
| gNCA | circular | alien_monster+crab+lizard+microbe+rooster | 0 | 0.001669 | 14961 | 20261017T172959Z-cfg24aa8602e7e9-4d75011e |
| gNCA | replicate | alien_monster+crab | 1 | 0.001825 | 15666 | 20261016T182526Z-cfgac2b9bf5e9a2-3db557f0 |
| gNCA | replicate | alien_monster+lizard | 0 | 0.001065 | 14292 | 20261017T171734Z-cfg0718fb77e430-319d740c |
| gNCA | replicate | alien_monster+microbe | 0 | 0.001706 | 10180 | 20261015T235049Z-cfg8d53e3bb0d65-d39c7e75 |
| gNCA | replicate | alien_monster+microbe | 1 | 0.001318 | 11804 | 20261015T130824Z-cfgff807af7cbe8-34ebcc1c |
| gNCA | replicate | alien_monster+rooster | 0 | 0.007873 | 1086 | 20261016T080413Z-cfg6a624a515325-0d5ce4e1 |
| gNCA | replicate | alien_monster+rooster | 1 | 0.001564 | 15447 | 20261014T085056Z-cfg5c9c816e906d-b320a895 |
| gNCA | replicate | crab+lizard | 0 | 0.001591 | 15975 | 20261013T175010Z-cfg8832faa67f92-1076d399 |
| gNCA | replicate | crab+lizard | 1 | 0.001508 | 15776 | 20261017T041729Z-cfgc840bda0cb07-497468ad |
| gNCA | replicate | crab+microbe | 0 | 0.001692 | 14345 | 20261016T081924Z-cfg960d2bafa18e-5cc7b82d |
| gNCA | replicate | crab+rooster | 0 | 0.001138 | 15344 | 20261017T091529Z-cfgde58355ab00a-ad24b92e |
| gNCA | replicate | crab+rooster | 1 | 0.001302 | 15003 | 20261017T132005Z-cfgb9ef9a3b4d21-1a17aa3e |
| gNCA | replicate | lizard+microbe | 0 | 0.001718 | 14908 | 20261016T102306Z-cfgb72619fa01e4-50254f0d |
| gNCA | replicate | lizard+microbe | 1 | 0.001646 | 15217 | 20261018T001534Z-cfgefbd9de808ae-086ba500 |
| gNCA | replicate | lizard+rooster | 0 | 0.001164 | 15777 | 20261015T103339Z-cfg0f95e8c198ee-878f834f |
| gNCA | replicate | lizard+rooster | 1 | 0.001197 | 14772 | 20261013T212423Z-cfg7221b18f025b-e2bf9b7b |
| gNCA | replicate | microbe+rooster | 0 | 0.001374 | 15849 | 20261016T053651Z-cfge5a7fd1b257f-10afc82c |
| gNCA | replicate | microbe+rooster | 1 | 0.001797 | 15576 | 20261018T071431Z-cfg4d7d50d3a14c-6ba248fe |
| gNCA | replicate | alien_monster+crab+lizard | 0 | 0.001612 | 15728 | 20261017T115535Z-cfgc4b6be967373-75df5a14 |
| gNCA | replicate | alien_monster+crab+lizard | 1 | 0.001639 | 15379 | 20261016T202627Z-cfgffea9bd0f0a0-02b2ed1c |
| gNCA | replicate | alien_monster+crab+microbe | 0 | 0.001646 | 11847 | 20261014T230116Z-cfg12d8c81d02be-ba124739 |
| gNCA | replicate | alien_monster+crab+microbe | 1 | 0.002022 | 15393 | 20261015T051637Z-cfg04228ef91126-7c9d6363 |
| gNCA | replicate | alien_monster+crab+rooster | 0 | 0.001431 | 12864 | 20261017T093330Z-cfg50ad503f6ddd-4b8202a8 |
| gNCA | replicate | alien_monster+lizard+microbe | 1 | 0.001606 | 15300 | 20261014T131605Z-cfga0e39263d430-57188017 |
| gNCA | replicate | alien_monster+lizard+rooster | 0 | 0.001632 | 15653 | 20261015T055639Z-cfg6fa8e2592e17-8fa7de57 |
| gNCA | replicate | alien_monster+microbe+rooster | 0 | 0.002166 | 15166 | 20261014T173525Z-cfg90563dd98740-0b878bca |
| gNCA | replicate | alien_monster+microbe+rooster | 1 | 0.001637 | 15983 | 20261016T041040Z-cfgc3dee72b3c59-dddc1b1e |
| gNCA | replicate | crab+lizard+microbe | 0 | 0.00172 | 15711 | 20261014T142104Z-cfg07d471ac12ce-da73fcde |
| gNCA | replicate | crab+lizard+microbe | 1 | 0.001711 | 12175 | 20261017T122603Z-cfgbe7356b474d2-18a833d7 |
| gNCA | replicate | crab+lizard+rooster | 0 | 0.001473 | 15913 | 20261015T060935Z-cfg2f0d37bbb23b-4272f897 |
| gNCA | replicate | crab+lizard+rooster | 1 | 0.00165 | 15913 | 20261016T035120Z-cfg27ae59be6fe6-213ca5df |
| gNCA | replicate | crab+microbe+rooster | 0 | 0.001556 | 15624 | 20261014T200249Z-cfge5098cecbda0-9a2a7a39 |
| gNCA | replicate | crab+microbe+rooster | 1 | 0.001743 | 15326 | 20261014T223339Z-cfg71b4becfe7b1-65786d93 |
| gNCA | replicate | lizard+microbe+rooster | 0 | 0.001607 | 15417 | 20261015T034337Z-cfgb258465d1bee-1078d2d7 |
| gNCA | replicate | lizard+microbe+rooster | 1 | 0.002142 | 15854 | 20261017T074028Z-cfg70e04fd4668c-3d213911 |
| gNCA | replicate | alien_monster+crab+lizard+microbe | 0 | 0.001658 | 14430 | 20261018T021426Z-cfg62a89c8f7968-5f9a8ce1 |
| gNCA | replicate | alien_monster+crab+lizard+microbe | 1 | 0.002132 | 14916 | 20261016T220333Z-cfg12c999e71eda-6ef219e1 |
| gNCA | replicate | alien_monster+crab+lizard+rooster | 0 | 0.006921 | 2993 | 20261015T161410Z-cfg3cf4128a7b61-7cd29e07 |
| gNCA | replicate | alien_monster+crab+lizard+rooster | 1 | 0.002425 | 15897 | 20261013T161451Z-cfg08dfe767869c-92060983 |
| gNCA | replicate | alien_monster+crab+microbe+rooster | 0 | 0.002687 | 15893 | 20261016T225336Z-cfg04fce4b65c50-63f679d4 |
| gNCA | replicate | alien_monster+crab+microbe+rooster | 1 | 0.001981 | 13017 | 20261015T130833Z-cfgf86051893c45-135a9595 |
| gNCA | replicate | alien_monster+lizard+microbe+rooster | 0 | 0.001868 | 15860 | 20261015T195632Z-cfgf6c857d3bae4-c39ea2da |
| gNCA | replicate | alien_monster+lizard+microbe+rooster | 1 | 0.001563 | 15937 | 20261013T171354Z-cfg75b2fd01d98a-a5964bc3 |
| gNCA | replicate | crab+lizard+microbe+rooster | 0 | 0.002636 | 14605 | 20261017T192427Z-cfg26ccf200e227-a3a43d8d |
| gNCA | replicate | crab+lizard+microbe+rooster | 1 | 0.005972 | 3867 | 20261013T210013Z-cfg9c161ca8a8f2-6fb2780f |
| gNCA | replicate | alien_monster+crab+lizard+microbe+rooster | 1 | 0.004117 | 15885 | 20261017T224705Z-cfgc49a71f671c5-76d8c5e6 |
