
# KLUE_Relation_Extraction

- [Relation Extraction](#relation-extraction)
  - [Task](#Task)
- [Model](#model)
- [Data](#data)
  - [train dataset form](#train-dataset-form)
  - [Preprocessing](#preprocessing)
    - [Relation classes](#relation-classes)
  - [Test dataset](#test-dataset)
- [How to use](#how-to-use)
  - [Install requirements](#install-requirements)
  - [Train](#train)
  - [Inference](#inference)
  - [Evaluation](#evaluation)
- [License](#license)

---

## Relation Extraction
Relation Extraction(관계 추출)은 문장의 단어(Entity)에 대한 속성과 관계를 예측하는 문제입니다. 관계 추출은 지식 그래프 구축을 위한 핵심 구성 요소로, 구조화된 검색, 감정 분석, 질문 답변하기, 요약과 같은 자연어처리 응용 프로그램에서 중요합니다. 비구조적인 자연어 문장에서 구조적인 triple을 추출해 정보를 요약하고, 중요한 성분을 핵심적으로 파악할 수 있습니다.
#### Task
본 프로젝트에서는 문장 내 두 단어의 relation이 labeling된 데이터를 이용하여 model을 학습하고 임의의 문장에 대해 문장 내 두 단어의 relation을 예측했습니다.  

## Model
본 프로젝트에서는 🤗huggingface [XLM-RoBERTa](https://huggingface.co/transformers/model_doc/xlmroberta.html) 의 large model을 사용하였습니다.

## Data
### train dataset form
9000개의 문장 데이터 사용 (private, .tsv)

![Raw dataset](https://user-images.githubusercontent.com/77161691/115668486-1fd41200-a382-11eb-950e-ad1d1340f769.png)

* column 1: 데이터가 수집된 정보
* column 2: sentence
* column 3: entity 1
* column 4: entity 1의 시작 지점
* column 5: entity 1의 끝 지점
* column 6: entity 2
* column 7: entity 2의 시작 지점
* column 8: entity 2의 끝 지점
* column 9: relation

### Preprocessing
train data는 Raw data의 sentence, entity1, entity2, relation column 사용 (load_data.py)

#### Relation classes
42개의 class 설정
```
with open('./label_type.pkl', 'rb') as f:
    label_type = pickle.load(f)

{'관계_없음': 0, '인물:배우자': 1, '인물:직업/직함': 2, '단체:모회사': 3, '인물:소속단체': 4, '인물:동료': 5, '단체:별칭': 6, '인물:출신성분/국적': 7, '인물:부모님': 8, '단체:본사_국가': 9, '단체:구성원': 10, '인물:기타_친족': 11, '단체:창립자': 12, '단체:주주': 13, '인물:사망_일시': 14, '단체:상위_단체': 15, '단체:본사_주(도)': 16, '단체:제작': 17, '인물:사망_원인': 18, '인물:출생_도시': 19, '단체:본사_도시': 20, '인물:자녀': 21, '인물:제작': 22, '단체:하위_단체': 23, '인물:별칭': 24, '인물:형제/자매/남매': 25, '인물:출생_국가': 26, '인물:출생_일시': 27, '단체:구성원_수': 28, '단체:자회사': 29, '인물:거주_주(도)': 30, '단체:해산일': 31, '인물:거주_도시': 32, '단체:창립일': 33, '인물:종교': 34, '인물:거주_국가': 35, '인물:용의자': 36, '인물:사망_도시': 37, '단체:정치/종교성향': 38, '인물:학교': 39, '인물:사망_국가': 40, '인물:나이': 41} 
```

### Test dataset
1000개의 문장 데이터 사용 (private, .tsv)

![Raw dataset](https://user-images.githubusercontent.com/77161691/115676003-1f3f7980-a38a-11eb-9f3d-23b772b19fd2.png)

* column 1-8: raw train dataset과 동일
* column 9: 'blind' relation -> will be predicted

## How to use
### Install requirements
```
pip install -r requirements.txt
```

### Train
```
python train.py
```
Please refer to train.py for train arguments. (Optional)

### Inference
```
python inference.py
```
Please refer to inference.py for inference arguments. (Optional)

### evaluation
```
python eval_acc.py
```

## License
CC-BY-SA
