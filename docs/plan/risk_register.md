# Risk Register

## Risk 1 — Dataset download / storage becomes too heavy
Impact: high  
Mitigation:
- start with one small smoke subset
- save a manifest of selected category files
- do not download more categories than needed

## Risk 2 — Leakage through repeated products
Impact: high  
Mitigation:
- split by `parent_asin`
- validate split integrity with a dedicated script

## Risk 3 — Notebook spaghetti
Impact: medium  
Mitigation:
- move reusable code into `src/`
- notebooks call functions, not the other way around

## Risk 4 — Baseline appears too weak / unfair
Impact: high  
Mitigation:
- keep TF-IDF strong
- use sensible vocabulary limits
- report both sparse and dense classical baselines

## Risk 5 — Word2Vec training quality is unstable
Impact: medium  
Mitigation:
- fix seeds where possible
- track vocabulary size and token statistics
- start with one stable config before ablations

## Risk 6 — Team scope creep
Impact: high  
Mitigation:
- freeze scope in week 1
- treat all extra ideas as stretch goals only
