# Branchseed — one setup command, one run command (the brief asks for exactly this)
PY ?= python3
DATA ?= data
REFS ?= refs

.PHONY: help setup selftest inspect run score clean submission

help:
	@echo "make setup       install dependencies"
	@echo "make selftest    build a synthetic case, detect, and score it end to end"
	@echo "make inspect     characterise every case in \$$DATA"
	@echo "make run         run the detector over \$$DATA -> preds/ and viz/"
	@echo "make score       score preds/ against \$$REFS"
	@echo "make submission  assemble submission/ for upload"

setup:
	$(PY) -m pip install -r requirements.txt

selftest:
	$(PY) make_phantom.py --out phantom
	mkdir -p preds viz
	$(PY) run.py --image phantom/data/phantom01/orig1.nii.gz \
	             --aorta-mask phantom/data/phantom01/mask1.nii.gz \
	             --output preds/phantom01.json --viz viz/phantom01.png
	$(PY) evaluate.py --pred preds/phantom01.json --ref phantom/refs/ --match-mm 3,5

inspect:
	$(PY) inspect_data.py --data $(DATA) --csv characterisation.csv

run:
	./run_all.sh $(DATA) $(REFS)

score:
	$(PY) evaluate.py --pred preds/ --ref $(REFS) --match-mm 3,5,8 --per-case

submission:
	rm -rf submission && mkdir -p submission/predictions submission/visual_checks
	cp run.py evaluate.py inspect_data.py make_phantom.py run_all.sh \
	   requirements.txt README.md Makefile submission/
	-cp preds/*.json submission/predictions/ 2>/dev/null
	-cp viz/*.png submission/visual_checks/ 2>/dev/null
	@echo "submission/ ready — zip it and upload"

clean:
	rm -rf preds viz phantom phantom_multi __pycache__ features.csv characterisation.csv
