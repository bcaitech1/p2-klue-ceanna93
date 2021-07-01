# Pstage_02_KLUE_Relation_extraction

### training
* python train.py

### inference
* python inference.py --model_dir=[model_path]
* ex) python inference.py --model_dir=./results/checkpoint-500

### evaluation
* python eval_acc.py

#### 파일 설명
- inference.py
    - 모델의 inference 결과를 submission 파일로 생성.
    - ensemble 함수와 단일 모델 inference 함수
        + ensemble을 위해서는 fine-tuning한 모델 사용(xlm-roberta, xlm-roberta, koelectra, bert) 없는 경우에 대한 예외 처리 없이 구현
        
- load_data.py
    - tokenize와 Dataset 정의
    - xlm-roberta와 koelectra 별 tokenize 방법이 달라 각 모델별 함수 구현
    - 앙상블을 위한 데이터셋 추가
    
- train.py
    - 모델을 train 하기 위한 코드
    - 다른 모델을 train할 때 코드 변경이 필요
    
- train_bert.py
    - inference에서 다른 모델과 함께 사용될 multilingual bert 모델을 학습시키기 위한 파일
