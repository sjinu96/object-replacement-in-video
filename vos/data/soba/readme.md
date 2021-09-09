# Preprocessing SOBA

### Download raw images and annotations

1. Download
2. Unzip
3. SOBA 폴더 & annotations 폴더
아래와 같은 폴더 상황 만들기
```bash
+--SOBA
|   +--ADE
|   +--COCO
|   +--SBU
|   +--SBU-test
|   +--WEB
+--annotations
|   +--SOBA_train_relation.json
|   +--SOBA_train.json
|   +--SOBA_val.json
# 밑은 github과 동일해서 디렉토리 내부는 표현 안함
+--common
+--dist
+--pysobatools
+--Makefile
+--par_crop.py
+--setup.py
+--readme.md
```

### c로 된 코드 사용할 수 있게 하는 작업
````shell
make  # Makefile 작동
````

### Crop & Generate data info

````shell
#python par_crop.py -h
python par_crop.py --enable_mask --num_threads 24
python gen_json.py
````
