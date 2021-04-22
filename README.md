# Pstage_03_KLUE_Relation_Extraction 
## With xml-Roberta-large

## Relation Extraction
Relation Extraction(관계 추출)은 문장의 단어(Entity)에 대한 속성과 관계를 예측하는 문제입니다. 관계 추출은 지식 그래프 구축을 위한 핵심 구성 요소로, 구조화된 검색, 감정 분석, 질문 답변하기, 요약과 같은 자연어처리 응용 프로그램에서 중요합니다. 비구조적인 자연어 문장에서 구조적인 triple을 추출해 정보를 요약하고, 중요한 성분을 핵심적으로 파악할 수 있습니다.

## Data
### input: sentence, entity1, entity2 의 정보를 입력으로 사용 합니다.
* sentence: 오라클(구 썬 마이크로시스템즈)에서 제공하는 자바 가상 머신 말고도 각 운영 체제 개발사가 제공하는 자바 가상 머신 및 오픈소스로 개발된 구형 버전의 온전한 자바 VM도 있으며, GNU의 GCJ나 아파치 소프트웨어 재단(ASF: Apache Software Foundation)의 하모니(Harmony)와 같은 아직은 완전하지 않지만 지속적인 오픈 소스 자바 가상 머신도 존재한다.
* entity 1: 썬 마이크로시스템즈
* entity 2: 오라클
* relation: 단체:별칭

### output: 42개 relation classes 중 1개의 class를 예측한 값입니다.

## Usage
### Install requirements
```
pip install -r requirements.txt
```

### Train
```
python train.py + [train arguments]
```
Please refer to train.py for train arguments. (Optional)

### Inference
```
python inference.py + [train conditions]
```
Please refer to inference.py for inference arguments. (Optional)

### evaluation
```
* python eval_acc.py + [train conditions]
```

## License
CC-BY-SA
